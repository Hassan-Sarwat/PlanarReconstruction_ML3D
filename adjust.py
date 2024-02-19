
import os
import random

# This file is used to get a subsample of the train data. Auxillary file and not important to overall experiment

# Replace 'your_directory_path' with the path to your directory containing NZP files
directory_path = '/cluster/52/sarwath/snet/output/processed/train/'

# Get the list of all files in the directory
all_files = os.listdir(directory_path)
all_files = [i for i in all_files if i[-3:]=='npz']

# Ensure you have enough files for the required split
total_samples = 100
if len(all_files) < total_samples:
    raise ValueError(f"Not enough files in the directory. Expected at least {total_samples} files.")

# # Randomly choose 10,000 files for training and 2,000 for validation
train_files = random.sample(all_files, 100)
# # val_files = random.sample(set(all_files) - set(train_files), 2000)

# Write the training file names to a new text file
train_output_file_path = '/cluster/52/sarwath/snet/output/processed/train_cont.txt'
with open(train_output_file_path, 'w') as train_output_file:
    for file_name in train_files:
        train_output_file.write(file_name + '\n')

