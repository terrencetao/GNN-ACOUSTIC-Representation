import tensorflow as tf
import os
import numpy as np
import pickle
from dtaidistance import dtw
from scipy.spatial.distance import euclidean
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse 




import numpy as np
import dgl
import torch
import pickle 
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure reproducibility by setting a seed (optional)
np.random.seed(42)

# Function to extract spectrograms from the dataset and squeeze them
def extract_spectrograms(dataset):
    spectrograms = []
    labels = []
    for spectrogram_batch, label_batch in dataset:
        for spectrogram in spectrogram_batch:
            spectrograms.append(tf.squeeze(spectrogram, axis=-1).numpy())
        for label in label_batch:
            labels.append(label.numpy())
    return spectrograms, labels



# Function to compute DTW distance between two spectrograms using dtaidistance
def compute_dtw_distance(spectrogram1, spectrogram2):
    distances = []
    for k in range(spectrogram1.shape[0]):  # iterate over frequency bins
        d = dtw.distance(spectrogram1[k, :], spectrogram2[k, :])
        distances.append(d)
    return np.mean(distances)

# Wrapper function for parallel processing
def compute_distance_for_pair(spectrograms, i, j):
    distance = compute_dtw_distance(spectrograms[i], spectrograms[j])
    return i, j, distance

def compute_dtw_similarity_matrix(spectrograms):
    num_spectrograms = len(spectrograms)
    similarity_matrix = np.zeros((num_spectrograms, num_spectrograms))
    
    # Create a list of pairs (i, j) for the upper triangle excluding the diagonal
    pairs = [(i, j) for i in range(num_spectrograms) for j in range(i + 1, num_spectrograms)]
    
    # Use Parallel and delayed to parallelize the computation
    results = Parallel(n_jobs=-1)(delayed(compute_distance_for_pair)(spectrograms, i, j) for i, j in tqdm(pairs))
    
    # Fill in the similarity matrix for the upper triangle and mirror it for the lower triangle
    for i, j, distance in results:
        similarity_matrix[i, j] = np.exp(-distance)  # Convert distance to similarity
    
    # Mirror the upper triangle to the lower triangle to ensure symmetry
    similarity_matrix = similarity_matrix + similarity_matrix.T
    
    return similarity_matrix
    
# Function to compute the median distance for each label group
def compute_median_distances(similarity_matrix, labels):
    unique_labels = np.unique(labels)
    median_distances = {}

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        distances = []

        for i in indices:
            for j in indices:
                if i != j:
                    distances.append(similarity_matrix[i, j])

        median_distances[label] = np.median(distances)

    return median_distances
    
def compute_iqr_thresholds(similarity_matrix, labels):
    unique_labels = np.unique(labels)
    iqr_thresholds = {}

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        distances = []

        for i in indices:
            for j in indices:
                if i != j:
                    distances.append(similarity_matrix[i, j])

        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        iqr_thresholds[label] = q1 - 1.5 * iqr  # Threshold for outliers

    return iqr_thresholds
    
def sim_matrix(method,subset_labels=None, subset_spectrograms=None):
   if method == 'dtw':
      similarity_matrix = compute_dtw_similarity_matrix(subset_spectrograms)
   elif method == 'fixed':
   # Convert labels list to a NumPy array
      labels_train_np = np.copy(subset_labels)

# Create a comparison matrix
      comparison_matrix = labels_train_np[:, None] == labels_train_np[None, :]

# Convert boolean matrix to integer matrix (0 and 1)
      similarity_matrix = comparison_matrix.astype(int)

   return similarity_matrix
   
   
# Function to perform stratified sampling
def stratified_sample(spectrograms, labels, subset_size):
    if subset_size > len(labels):
       return spectrograms,labels
    # Convert labels to a numpy array if not already
    labels = np.array(labels)
    
    # Perform stratified sampling using train_test_split
    stratified_spectrograms, _, stratified_labels, _ = train_test_split(
        spectrograms, labels, 
        train_size=subset_size, 
        stratify=labels,
        random_state=42  # for reproducibility
    )
    
    return stratified_spectrograms, stratified_labels
    

if __name__ == "__main__":
	logging.info(f' ----------------------------------------------------- BUILD GRAPH STEP  -----------------------------------------------')
	parser = argparse.ArgumentParser()
	  
	parser.add_argument('--sub_units', help='fraction of data', required=True)    
	parser.add_argument('--method', help='', required=True)
	parser.add_argument('--dataset', help='name of dataset', required=True)
	parser.add_argument('--base_dir', help='feature type: spec or mfcc', required=True)
        
	args = parser.parse_args()
	sub_units = int(args.sub_units)    
	 
	# Define the directory where datasets are saved
	data_dir = os.path.join(args.base_dir,'saved_datasets',args.dataset)
	# Define the directory to save the datasets
	save_dir = os.path.join(args.base_dir,'saved_matrix',args.dataset, args.method)
	os.makedirs(save_dir, exist_ok=True)
	# File paths
	similarity_matrix_path = os.path.join(save_dir, f'similarity_matrix_with_labels_{sub_units}.npy')
	subset_spectrogram_path = os.path.join(save_dir, f'subset_spectrogram_{sub_units}.npy')
	subset_labels_path = os.path.join(save_dir, f'subset_label_{sub_units}.npy')
	subset_val_spectrogram_path = os.path.join(save_dir, f'subset_val_spectrogram_{sub_units}.npy')
	subset_val_labels_path = os.path.join(save_dir, f'subset_val_label_{sub_units}.npy')

    # Check if the similarity matrix file already exists
	if not os.path.isfile(similarity_matrix_path):
		# Load the datasets
		loaded_train_spectrogram_ds = tf.data.experimental.load(os.path.join(data_dir, 'train_spectrogram_ds'))
		loaded_val_spectrogram_ds = tf.data.experimental.load(os.path.join(data_dir, 'val_spectrogram_ds'))
		#loaded_test_spectrogram_ds = tf.data.experimental.load(os.path.join(data_dir, 'test_spectrogram_ds'))

		print("Datasets loaded successfully.")


		# Extract spectrograms
		train_spectrograms, labels_train = extract_spectrograms(loaded_train_spectrogram_ds)
		val_spectrograms, labels_val = extract_spectrograms(loaded_val_spectrogram_ds)
		#test_spectrograms, labels_test = extract_spectrograms(loaded_test_spectrogram_ds)

		

		# Set your desired subset size
		subset_size = sub_units

		# Perform stratified sampling for training and validation sets
		#subset_spectrograms, subset_labels = stratified_sample(train_spectrograms, labels_train, subset_size)
		subset_spectrograms, subset_labels = train_spectrograms, labels_train
		#subset_val_spectrograms, subset_val_labels = stratified_sample(val_spectrograms, labels_val, subset_size)
		subset_val_spectrograms, subset_val_labels = val_spectrograms, labels_val


		# Proceed with the rest of your code as before
		# Convert lists to numpy arrays if needed
		subset_spectrograms = np.array(subset_spectrograms)
		subset_val_spectrograms = np.array(subset_val_spectrograms)
		subset_labels = np.array(subset_labels)
		subset_val_labels = np.array(subset_val_labels)
			
			# Calculate total size and size for each label
		total_size = len(subset_labels)
		label_counts = np.unique(subset_labels, return_counts=True)
		label_sizes = dict(zip(label_counts[0], label_counts[1]))
		
		total_val_size = len(subset_val_labels)
		label_val_counts = np.unique(subset_val_labels, return_counts=True)
		label_val_sizes = dict(zip(label_val_counts[0], label_val_counts[1]))
		# Convert label sizes to a DataFrame
		df = pd.DataFrame([label_sizes], index=['size'])
		df_val = pd.DataFrame([label_val_sizes], index=['size'])

		# Add the total size as a separate row
		df['Total'] = total_size
		df_val['Total'] = total_val_size
		
		# Define the CSV file path
		csv_file_path = os.path.join(save_dir,f'sample_size_info_{sub_units}.csv')
		csv_val_file_path = os.path.join(save_dir,f'sample_size_val_info_{sub_units}.csv')
		# Check if the file exists
		file_exists = os.path.isfile(csv_file_path)

		# Save to CSV
		if not file_exists:
		    # If the file doesn't exist, create it with headers
		    df.to_csv(csv_file_path, mode='w', header=True, index=False)
		else:
		    # If the file exists, append without headers
		    df.to_csv(csv_file_path, mode='a', header=False, index=False)

		# Check if the file exists
		file_val_exists = os.path.isfile(csv_val_file_path)

		# Save to CSV
		if not file_val_exists:
		    # If the file doesn't exist, create it with headers
		    df_val.to_csv(csv_val_file_path, mode='w', header=True, index=False)
		else:
		    # If the file exists, append without headers
		    df_val.to_csv(csv_val_file_path, mode='a', header=False, index=False)
		# Compute the median distances for each label group
		#median_distances = compute_median_distances(similarity_matrix, subset_labels)
		#bornes_inferieures_iqr = compute_iqr_thresholds(similarity_matrix, subset_labels)
		# Filter the similarity matrix based on the median thresholds and set diagonal to zero

		#medianes = np.array(list(median_distances.values()))
		#nan_mask = np.isnan(medianes)
		#filtered_similarity_matrix = filter_similarity_matrix(similarity_matrix, subset_labels, threshold=int(args.ta), k=int(args.num_n))

		similarity_matrix = sim_matrix(method=args.method,  subset_labels=subset_labels, subset_spectrograms=subset_spectrograms)

		print(similarity_matrix)
		# Append labels as an additional column
		matrix_with_labels = np.hstack((subset_labels[:, np.newaxis], similarity_matrix))
		
		
		# Save the matrix with labels and other data
		np.save(similarity_matrix_path, matrix_with_labels)
		np.save(subset_spectrogram_path, subset_spectrograms)
		np.save(subset_labels_path, subset_labels)
		np.save(subset_val_spectrogram_path, subset_val_spectrograms)
		np.save(subset_val_labels_path, subset_val_labels)


		print("Acoustic similarity matrix computed successfully.")
	else:
		print(f"File {similarity_matrix_path} already exists. Skipping computation.")
	    

