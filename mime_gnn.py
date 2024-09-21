import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CNN Model Definition
class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (input_shape[1] // 4) * (input_shape[2] // 4), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        embeddings = F.relu(self.fc3(x))
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x, embeddings
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train_cnn(model, train_loader,criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels, gnn_embeddings in train_loader:
            optimizer.zero_grad()
            outputs, embeddings = model(inputs)
            loss = criterion(outputs, labels) + F.mse_loss(gnn_embeddings,embeddings)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

def evaluate_cnn(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_embeddings = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs, embeddings = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Accumulate embeddings
            all_embeddings.append(embeddings)

    # Concatenate embeddings from all batches
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    return correct / total, all_embeddings
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--epochs', help='number of epochs', required=True)
   parser.add_argument('--dataset', help='name of dataset', required=False)
   parser.add_argument('--mma', help='', required=True)
   parser.add_argument('--msa', help='method to compute heterogeneous similarity', required=True)
   parser.add_argument('--sub_units', help='fraction of data', required=True)  
   parser.add_argument('--base_dir', help='feature type: spec or mfcc', required=True)
     
   args = parser.parse_args()
   sub_units = args.sub_units
   
   embedding_folder = os.path.join(args.base_dir,'saved_embeddings', args.dataset, args.mma, args.msa, f'sub_units_{args.sub_units}')
   matrix_dir = os.path.join('saved_matrix',args.dataset, args.mma)
   labels_np = np.load(os.path.join(matrix_dir,f'subset_label_{sub_units}.npy'))
   val_labels_np = np.load(os.path.join(matrix_dir, f'subset_val_label_{sub_units}.npy'))
   subset_val_spectrograms = np.load(os.path.join(matrix_dir, f'subset_val_spectrogram_{sub_units}.npy'))
   spectrograms = np.load(os.path.join(matrix_dir,f'subset_spectrogram_{sub_units}.npy')) 
   gnn_embeddings = np.load(os.path.join(embedding_folder, f'hibrid_node_embeddings_{args.sub_units}.npy'))
   num_classes = len(np.unique(labels_np))
   
# Prepare data for CNN
   gnn_embeddings_tensor = torch.tensor(gnn_embeddings, dtype=torch.float32)
   spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32)
   val_spectrograms_tensor = torch.tensor(subset_val_spectrograms, dtype=torch.float32)
   labels_tensor = torch.tensor(labels_np, dtype=torch.long)
   val_labels_tensor = torch.tensor(val_labels_np, dtype=torch.long)
   spectrograms_tensor = spectrograms_tensor.unsqueeze(1) 
   val_spectrograms_tensor = val_spectrograms_tensor.unsqueeze(1)
#X_train, X_test, y_train, y_test = train_test_split(spectrograms_tensor, labels_tensor, test_size=0.2, random_state=42)
   X_train, X_test, y_train, y_test = spectrograms_tensor, val_spectrograms_tensor, labels_tensor, val_labels_tensor
   train_loader = DataLoader(TensorDataset(X_train, y_train, gnn_embeddings_tensor), batch_size=32, shuffle=True)
   test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# Define and train the CNN model
   logging.info(f'train the CNN model')
   input_shape = spectrograms_tensor.shape[1:]  # (1, height, width)
   cnn_model = SimpleCNN(input_shape, num_classes)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
   train_cnn(cnn_model, train_loader, criterion, optimizer, num_epochs=int(args.epochs),)
   accuracy_cnn, mime_embeddings_hibrid = evaluate_cnn(cnn_model, test_loader)
   logging.info(f'CNN Model Accuracy: {accuracy_cnn}')
   np.save(os.path.join(embedding_folder, f'mime_hibrid_node_val_embeddings_{args.sub_units}.npy'), mime_embeddings_hibrid)
   train_loader_eval = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
   accuracy_train, mime_embeddings_hibrid_train = evaluate_cnn(cnn_model, train_loader_eval)
   np.save(os.path.join(embedding_folder, f'mime_hibrid_node_embeddings_{args.sub_units}.npy'), mime_embeddings_hibrid_train)
   logging.info(f'CNN Model Accuracy: {accuracy_train}')
   torch.save(cnn_model,'models/mime.pth')
