import os
import torch
import dgl
import numpy as np
import csv

from gnn_model import GCN
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from generate_similarity_matrix_acoustic import compute_distance_for_pair,  distance_dtw
import math
import logging
from joblib import Parallel, delayed
import tensorflow as tf
from weak_ML2 import SimpleCNN
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_graphs(path):
    return dgl.load_graphs(path)

def ml_distance(ml_model, spec1, spec2):
    ml_model.eval()
    # Ensure the input spectrograms are in the correct shape for the model
    spec1 = torch.tensor(spec1).unsqueeze(0)  # Add batch dimension
    spec2 = torch.tensor(spec2).unsqueeze(0)  # Add batch dimension
    
    # Stack the spectrograms into a batch
    batch = torch.stack([spec1, spec2], dim=0)
    
    
    
    # Pass the batch through the model
    with torch.no_grad():  # Disable gradient calculation
        logits = ml_model(batch)
        ml_predictions = F.softmax(logits, dim=1)
    
    # Calculate the KL divergence (or another distance measure) between the predictions
    kl_divergence = F.kl_div(ml_predictions[0].log(), ml_predictions[1], reduction='batchmean')
    
    return kl_divergence   



def add_new_nodes_to_graph_knn(dgl_G, new_node_spectrograms, k, distance_function,ml, n_jobs=-1):
    num_existing_nodes = dgl_G.number_of_nodes()
    num_new_nodes = new_node_spectrograms.shape[0]
    
    # Add new nodes to the graph
    dgl_G.add_nodes(num_new_nodes)

    # Add features for the new nodes
    new_features = torch.from_numpy(new_node_spectrograms)
    dgl_G.ndata['feat'][num_existing_nodes:num_existing_nodes + num_new_nodes] = new_features

    existing_features = dgl_G.ndata['feat'][:num_existing_nodes].numpy()

    def process_new_node(new_node_index):
        new_node_spectrogram = new_node_spectrograms[new_node_index - num_existing_nodes]
        
        # Compute distances to all existing nodes
        distances = [distance_function(ml,new_node_spectrogram, existing_features[i]) for i in range(num_existing_nodes)]
        
        # Select the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        
        edges = []
        for i in nearest_indices:
            distance = distances[i]
            similarity = np.exp(-distance)
            edges.append((new_node_index, i, similarity))
            edges.append((i, new_node_index, similarity))
        return edges

    # Use joblib to parallelize the processing of new nodes
    all_edges = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index) for new_node_index in tqdm(range(num_existing_nodes, num_existing_nodes + num_new_nodes)))
    
    # Flatten the list of edges and add them to the graph
    src_nodes, dst_nodes, weights = zip(*[(src, dst, weight) for edges in all_edges for src, dst, weight in edges])
    dgl_G.add_edges(src_nodes, dst_nodes, {'weight': torch.tensor(weights, dtype=torch.float32)})

    return dgl_G, num_existing_nodes    
    

    
    
def generate_embeddings(gcn_model, dgl_G,num_existing_nodes, new_node_spectrograms):
    """
    Generate embeddings for new nodes using the provided GCN model.
    
    Parameters:
    gcn_model (torch.nn.Module): The trained GCN model.
    dgl_G (dgl.DGLGraph): The graph structure containing existing nodes.
    new_node_spectrograms (np.ndarray): Spectrograms of the new nodes to be added.
    k (int): The number of neighbors to connect each new node to.
    compute_distance (function): A function to compute the distance between nodes.
    
    Returns:
    np.ndarray: Embeddings for the new nodes.
    """
    # Add new nodes to the graph and get the updated graph and the number of existing nodes
    #dgl_G, num_existing_nodes = add_new_nodes_to_graph_randomly(dgl_G, new_node_spectrograms, k, compute_distance)
    
    # Extract edge weights and features from the graph
    edge_weights = dgl_G.edata['weight']
    features = dgl_G.ndata['feat']
    
    # Generate embeddings using the GCN model
    with torch.no_grad():
        gcn_model.eval()
        _,embeddings = gcn_model(dgl_G, features, edge_weights)
        embeddings = embeddings.numpy()
    
    # Extract the embeddings for the new nodes
    num_new_nodes = new_node_spectrograms.shape[0]
    new_node_indices = np.arange(num_existing_nodes, num_existing_nodes + num_new_nodes)
    new_node_embeddings = embeddings[new_node_indices]
    
    # Debugging prints
    #print(embeddings.shape)
    #print(new_node_indices)
    
    return embeddings[:num_existing_nodes], new_node_embeddings



    




# Spectrogram baseline embeddings
def flatten_spectrograms(spectrograms):
    num_samples = spectrograms.shape[0]
    flattened_spectrograms = spectrograms.reshape(num_samples, -1)
    return flattened_spectrograms
    
    
logging.info(f' ----------------------------------------------------- Evaluation of Representation  -----------------------------------------------')
       
parser = argparse.ArgumentParser()


parser.add_argument('--num_n_a', help='method to compute a word similarity', required=True)
parser.add_argument('--ta', help='method to compute a word similarity', required=True)
parser.add_argument('--alpha', help='method to compute a word similarity', required=True)

parser.add_argument('--msa', help='method to compute heterogeneous similarity', required=True)

parser.add_argument('--mma', help='method to build acoustic matrix', required=True)
parser.add_argument('--drop_freq', help='dim frequency ', required=False)  
parser.add_argument('--drop_int', help='dim amplitude ', required=False) 
parser.add_argument('--sub_units', help='fraction of data', required=True)  
parser.add_argument('--dataset', help='name of dataset', required=True)
parser.add_argument('--base_dir', help='feature type: spec or mfcc', required=True)
args = parser.parse_args()


# Paths
graph_folder = os.path.join(args.base_dir,'saved_graphs',args.dataset,args.mma,args.msa)
model_folder = os.path.join(args.base_dir,'models')
matrix_folder = os.path.join(args.base_dir,'saved_matrix',args.dataset, args.mma)

# Create a new directory for saving embeddings
embedding_folder = os.path.join(args.base_dir,'saved_embeddings', args.dataset, args.mma, args.msa, f'sub_units_{args.sub_units}')
os.makedirs(embedding_folder, exist_ok=True)

# Load the homogeneous graph
glist, label_dict = load_graphs(os.path.join(graph_folder,f"kws_graph_{args.num_n_a}_{args.sub_units}.dgl"))
dgl_G = glist[0]

features = dgl_G.ndata['feat']
labels = dgl_G.ndata['label']
subset_val_labels = np.load(os.path.join(matrix_folder,f'subset_val_label_{args.sub_units}.npy'))
subset_val_spectrograms = np.load(os.path.join(matrix_folder,f'subset_val_spectrogram_{args.sub_units}.npy'))

# Define the input features size
in_feats = features[0].shape[0] * features[0].shape[1]
hidden_size = 512
num_classes = len(torch.unique(labels))
conv_param = [(1, 3, (20, 64)), 32, 2]
hidden_units = [32, 32]

# Load supervised GCN model
logging.info(f'Load supervised GCN model')
model_sup_path = os.path.join(model_folder, "gnn_model.pth")
loaded_model_sup = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
loaded_model_sup.load_state_dict(torch.load(model_sup_path))

kws_graph_path_val = os.path.join(graph_folder, f"kws_graph_val_{args.num_n_a}_{args.sub_units}.dgl")
acoustic_model =  torch.load('models/cnn.pth')
if not os.path.isfile(kws_graph_path_val):
  logging.info(f'Extract acoustic node representations from supervised GCN')
  dgl_G, num_existing_nodes = add_new_nodes_to_graph_knn(dgl_G, new_node_spectrograms=subset_val_spectrograms,  k=math.floor(int(args.num_n_a)/2), distance_function=ml_distance, ml=acoustic_model)
  
  kws_graph_path_val = os.path.join(graph_folder, f"kws_graph_val_{args.num_n_a}_{args.sub_units}.dgl")
  dgl.save_graphs(kws_graph_path_val, [dgl_G])
  print(f"dgl val save successfully")
else:
  print(f"File {kws_graph_path_val} already exists. Skipping computation.")
  num_existing_nodes = dgl_G.number_of_nodes()
  glist, label_dict= load_graphs(kws_graph_path_val)
  dgl_G = glist[0]
node_embeddings_sup, node_val_embeddings_sup = generate_embeddings(gcn_model=loaded_model_sup, 
                                                dgl_G=dgl_G,num_existing_nodes=num_existing_nodes, new_node_spectrograms=subset_val_spectrograms, 
                                               )



# Load unsupervised GCN model
logging.info(f'Load unsupervised GCN model')
model_unsup_path = os.path.join(model_folder, "gnn_model_unsup.pth")
loaded_model_unsup = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
loaded_model_unsup.load_state_dict(torch.load(model_unsup_path))

# Save the supervised embeddings to .npy files in the new path
np.save(os.path.join(embedding_folder, f'supervised_node_embeddings_{args.sub_units}.npy'), node_embeddings_sup)
np.save(os.path.join(embedding_folder, f'supervised_node_val_embeddings_{args.sub_units}.npy'), node_val_embeddings_sup)

# Load hibrid GCN model
logging.info(f'Load unsupervised GCN model')
model_hibrid_path = os.path.join(model_folder, "gnn_model_hibrid.pth")
loaded_model_hibrid = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
loaded_model_hibrid.load_state_dict(torch.load(model_hibrid_path))






# Extract labels for training
labels_np = labels.numpy()
val_labels_np = subset_val_labels




 
# Train and evaluate SVM for unsupervised embeddings
logging.info(f'Extract acoustic node representations From unsupervised GNN')
node_embeddings_unsup, node_val_embeddings_unsup = generate_embeddings(gcn_model=loaded_model_unsup, 
                                                dgl_G=dgl_G,num_existing_nodes=num_existing_nodes, new_node_spectrograms=subset_val_spectrograms, )

# Save the unsupervised embeddings to .npy files in the new path
np.save(os.path.join(embedding_folder, f'unsupervised_node_embeddings_{args.sub_units}.npy'), node_embeddings_unsup)
np.save(os.path.join(embedding_folder, f'unsupervised_node_val_embeddings_{args.sub_units}.npy'), node_val_embeddings_unsup)



logging.info(f'Extract acoustic node representations From unsupervised GNN')
node_embeddings_hibrid, node_val_embeddings_hibrid = generate_embeddings(gcn_model=loaded_model_hibrid, 
                                                dgl_G=dgl_G,num_existing_nodes=num_existing_nodes, new_node_spectrograms=subset_val_spectrograms, )
# Save the hibrid embeddings to .npy files in the new path
np.save(os.path.join(embedding_folder, f'hibrid_node_embeddings_{args.sub_units}.npy'), node_embeddings_hibrid)
np.save(os.path.join(embedding_folder, f'hibrid_node_val_embeddings_{args.sub_units}.npy'), node_val_embeddings_hibrid)


