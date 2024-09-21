from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse
from sklearn import svm
import logging 
import numpy as np
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to train and evaluate SVM and return accuracy
def train_evaluate_svm(X_train, X_test, y_train, y_test):
    # Define the parameter grid for 'kernel' and 'C'
    param_grid = {
        'kernel': ['sigmoid'],  # Possible kernels
        'C': [0.1, 1, 10, 100]  # Range of C values to try
    }

    # Initialize the SVM model
    clf = svm.SVC()

    # Set up GridSearchCV to find the best parameters
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')

    # Train the model with grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters from the grid search
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Use the best estimator to predict on test data
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return accuracy, best_params


    
# Spectrogram baseline embeddings
def flatten_spectrograms(spectrograms):
    num_samples = spectrograms.shape[0]
    flattened_spectrograms = spectrograms.reshape(num_samples, -1)
    return flattened_spectrograms
      
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
     
	logging.info(f'Train and evaluate SVM for spectrogram embeddings')
	
	embedding_folder = os.path.join(args.base_dir,'saved_embeddings', args.dataset, args.mma, args.msa, f'sub_units_{args.sub_units}')
	matrix_dir = os.path.join('saved_matrix',args.dataset, args.mma)
	labels_np = np.load(os.path.join(matrix_dir,f'subset_label_{sub_units}.npy'))
	val_labels_np = np.load(os.path.join(matrix_dir, f'subset_val_label_{sub_units}.npy'))
	subset_val_spectrograms = np.load(os.path.join(embedding_folder, f'mime_hibrid_node_val_embeddings_{args.sub_units}.npy'))
	spectrograms = np.load(os.path.join(embedding_folder, f'hibrid_node_embeddings_{args.sub_units}.npy')) 
	flattened_spectrograms = spectrograms
	flattened_val_spectrograms = subset_val_spectrograms
	print(flattened_val_spectrograms.shape)
	# Train and evaluate SVM for spectrogram embeddings
	accuracy_spectrogram = train_evaluate_svm( X_train=flattened_spectrograms, X_test=flattened_val_spectrograms, y_train=labels_np, y_test=val_labels_np)
	logging.info(f'SVM Model Accuracy: {accuracy_spectrogram}')
	accuracy_spectrogram_train = train_evaluate_svm( X_train=flattened_spectrograms, X_test=flattened_spectrograms, y_train=labels_np, y_test=labels_np)
	logging.info(f'SVM Model Accuracy train: {accuracy_spectrogram_train}')
