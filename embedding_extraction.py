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
from generate_similarity_matrix_acoustic import compute_distance_for_pair, compute_dtw_distance

import logging
from joblib import Parallel, delayed
import tensorflow as tf

from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_graphs(path):
    return dgl.load_graphs(path)
    

def add_new_nodes_to_graph_randomly(dgl_G, new_node_spectrograms, k, distance_function, n_jobs=-1):
    num_existing_nodes = dgl_G.number_of_nodes()
    num_new_nodes = new_node_spectrograms.shape[0]
    
    # Add new nodes to the graph
    dgl_G.add_nodes(num_new_nodes)

    # Add features for the new nodes
    new_features = torch.from_numpy(new_node_spectrograms)
    dgl_G.ndata['feat'][num_existing_nodes:num_existing_nodes + num_new_nodes] = new_features

    # Randomly select k nodes from the existing graph for each new node
    existing_indices = np.arange(num_existing_nodes)
    
    def process_new_node(new_node_index):
        selected_indices = np.random.choice(existing_indices, k, replace=False)
        new_node_spectrogram = new_node_spectrograms[new_node_index - num_existing_nodes]
        
        edges = []
        for i in selected_indices:
            distance = distance_function(new_node_spectrogram, dgl_G.ndata['feat'][i].numpy())
            similarity = np.exp(-distance)
            edges.append((new_node_index, i, similarity))
            edges.append((i, new_node_index, similarity))
        return edges

    # Use joblib to parallelize the processing of new nodes
    all_edges = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index) for new_node_index in tqdm(range(num_existing_nodes, num_existing_nodes + num_new_nodes)))
    
    # Flatten the list of edges and add them to the graph
    for edges in all_edges:
        for src, dst, weight in edges:
            dgl_G.add_edges(src, dst, {'weight': torch.tensor([weight], dtype=torch.float32)})
    
    return dgl_G, num_existing_nodes
    
    
def add_new_acoustic_nodes_to_hetero_graph(hetero_graph, new_node_spectrograms, k, distance_function, ml_model, threshold_probability, n_jobs=-1):
    num_existing_acoustic_nodes = hetero_graph.num_nodes('acoustic')
    num_existing_word_nodes = hetero_graph.num_nodes('word')
    num_new_nodes = new_node_spectrograms.shape[0]
    
    # Add new 'acoustic' nodes to the graph
    hetero_graph.add_nodes(num_new_nodes, ntype='acoustic')
    
    # Add features for the new 'acoustic' nodes
    flattened_spectrograms = flatten_spectrograms(new_node_spectrograms)
    new_features = torch.from_numpy(flattened_spectrograms)
    hetero_graph.nodes['acoustic'].data['feat'][num_existing_acoustic_nodes:num_existing_acoustic_nodes  + num_new_nodes] = new_features

    # Get softmax probabilities for connections to 'word' nodes using the ML model
    ml_model.eval()
    new_node_features = torch.tensor(new_node_spectrograms)
    new_node_features = new_node_features.view(new_node_features.shape[0],1 , new_node_features.shape[1], new_node_features.shape[2])
# Predict softmax probabilities
    with torch.no_grad():  # Disable gradient calculation
         logits = ml_model(new_node_features)
         ml_predictions = F.softmax(logits, dim=1)
         
    
    ml_probabilities = ml_predictions
    
    # Filter probabilities based on the threshold
    ml_probabilities = filter_similarity_matrix(ml_probabilities.numpy(), threshold=threshold_probability, k=k)
    
    def process_new_node(new_node_index):
        edges = []
        probabilities = []
        for word_node_index in range(num_existing_word_nodes):
            similarity = ml_probabilities[new_node_index - num_existing_acoustic_nodes, word_node_index]
            if similarity > 0:
                edges.append((new_node_index, word_node_index))
                probabilities.append(similarity)
        return edges, probabilities

    # Use joblib to parallelize the processing of new nodes
    all_edges_and_probs = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index) for new_node_index in tqdm(range(num_existing_acoustic_nodes, num_existing_acoustic_nodes + num_new_nodes)))
    
    # Flatten the list of edges and add them to the graph
    all_edges = [edge for edges, _ in all_edges_and_probs for edge in edges]
    all_probabilities = [prob for _, probs in all_edges_and_probs for prob in probs]
    
    if all_edges:
        src, dst = zip(*all_edges)
        src = torch.tensor(src, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)
        hetero_graph.add_edges(src, dst, etype=('acoustic', 'related_to', 'word'))
        new_weights = torch.tensor(all_probabilities, dtype=torch.float32)
        if 'weight' not in hetero_graph.edges['related_to'].data:
           num_existing_edges = hetero_graph.num_edges(('acoustic', 'related_to', 'word'))
           hetero_graph.edges['related_to'].data['weight'] = torch.zeros(num_existing_edges, dtype=torch.float32)
    
    # Assign new edge weights to the new edges only
        hetero_graph.edges['related_to'].data['weight'][-new_weights.shape[0]:] = new_weights

        #hetero_graph.edges['related_to'].data['weight'][num_existing_acoustic_nodes:num_existing_acoustic_nodes  + num_new_nodes] = torch.tensor(all_probabilities, dtype=torch.float32)
    return hetero_graph, num_existing_acoustic_nodes
    
    
    
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
        embeddings = gcn_model(dgl_G, features, edge_weights).numpy()
    
    # Extract the embeddings for the new nodes
    num_new_nodes = new_node_spectrograms.shape[0]
    new_node_indices = np.arange(num_existing_nodes, num_existing_nodes + num_new_nodes)
    new_node_embeddings = embeddings[new_node_indices]
    
    # Debugging prints
    #print(embeddings.shape)
    #print(new_node_indices)
    
    return embeddings[num_existing_nodes:], new_node_embeddings


def generate_embeddings_hetero(gcn_model, hetero_graph,num_existing_acoustic_nodes, new_node_spectrograms):
    """
    Generate embeddings for new nodes using the provided GCN model.
    
    Parameters:
    gcn_model (torch.nn.Module): The trained GCN model.
    hetero_graph (dgl.DGLHeteroGraph): The heterogeneous graph structure.
    new_node_spectrograms (np.ndarray): Spectrograms of the new nodes to be added.
    k (int): The number of neighbors to connect each new node to.
    compute_distance (function): A function to compute the distance between nodes.
    ml_model (tf.keras.Model): The ML model for predicting connection probabilities.
    threshold_probability (float): The threshold for filtering connection probabilities.
    n_jobs (int): The number of jobs for parallel processing.

    Returns:
    np.ndarray: Embeddings for the new nodes.
    """
    # Add new 'acoustic' nodes to the graph
    #hetero_graph, num_existing_acoustic_nodes = add_new_acoustic_nodes_to_hetero_graph(
    #    hetero_graph, 
    #    new_node_spectrograms, 
    #    k, 
    #    compute_distance, 
    #    ml_model, 
    #    threshold_probability, 
    #    n_jobs
    #)
    
    # Extract features from the graph
    features_dic = {
    'acoustic': hetero_graph.nodes['acoustic'].data['feat'],
    'word': hetero_graph.nodes['word'].data['feat']
}
    # Generate embeddings using the GCN model
    with torch.no_grad():
        gcn_model.eval()
        # Assuming the GCN model takes the graph and node features as input
        embeddings = gcn_model(hetero_graph, features_dic)
        embeddings = embeddings['acoustic'].numpy()
    
    # Extract the embeddings for the new nodes
    num_new_nodes = new_node_spectrograms.shape[0]
    new_node_indices = np.arange(num_existing_acoustic_nodes, num_existing_acoustic_nodes + num_new_nodes)
    new_node_embeddings = embeddings[new_node_indices]
    
    return embeddings[num_existing_acoustic_nodes:], new_node_embeddings
    




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
hidden_size = 64
num_classes = len(torch.unique(labels))
conv_param = [(1, 3, (20, 64)), 32, 2]
hidden_units = [32, 32]

# Load supervised GCN model
logging.info(f'Load supervised GCN model')
model_sup_path = os.path.join(model_folder, "gnn_model.pth")
loaded_model_sup = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
loaded_model_sup.load_state_dict(torch.load(model_sup_path))

logging.info(f'Extract acoustic node representations From supervised GNN')
dgl_G, num_existing_nodes = add_new_nodes_to_graph_randomly(dgl_G, new_node_spectrograms=subset_val_spectrograms,  k=int(args.num_n_a), distance_function=compute_dtw_distance)
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




