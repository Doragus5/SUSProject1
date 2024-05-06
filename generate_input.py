import os
import random

def generate_file_paths(directory, k, output_file):

    file_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    
    selected_file_paths = random.sample(file_paths, k)
    
    with open(output_file, 'w') as f:
        for file_path in selected_file_paths:
            f.write(file_path + '\n')

directory = '/home/students/inf/m/ml418309/SUS/SUSProject1/training_samples'
k = 5000 
output_file = 'input.txt'  

generate_file_paths(directory, k, output_file)