# Paths
# copy in the file path to where the hydromoth recordings are saved:
new_audio_dir = r"C:\Users\ben07\OneDrive - University College London\UCL PhD\Chapter 4 Bombs\spermonde2024\scratch\example_data" #<-------

# copy in the file path to where the results should be saved:
output_folder = r"C:\Users\ben07\OneDrive - University College London\UCL PhD\Chapter 4 Bombs\spermonde2024\scratch\example_output_folder" #<---------

# model directory
model_dir = r"C:\Users\ben07\OneDrive - University College London\UCL PhD\Chapter 4 Bombs\spermonde2024\archive\saved_model_v1"

# scratch folder
scratch_folder = r"C:\Users\ben07\OneDrive - University College London\UCL PhD\Chapter 4 Bombs\spermonde2024\scratch"

# batch size (number of files to run in one go, start at 100 but may need to reduce size if it fails)
batch_size = 25



############################################################################# 
# MASTER SCRIPT - DO NOT CHANGE ANYTHING BELOW HERE
# imports
import subprocess
import os
import datetime
from datetime import datetime
import numpy as np
import pandas as pd
import multiprocessing as mp
import sys

NUM_PROCESSES = 4
sample_rate = 8000

def multiproc_exec():
    proceed = mp.Event()
    processes = [mp.Process(target=run_batched_run_on_threads) for i in range(NUM_PROCESSES)]
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()
    

# Wrap in a function for multithreading.
def run_batched_run_on_threads():  #process_number: int):

    # create initial results csv
    results_table = pd.DataFrame(columns=['File','Timestamp'])
    results_table.to_csv(os.path.join(output_folder, 'temporary_results_table.csv'), index=False)

    # find the audio
    new_audio_files = [f for f in os.listdir(new_audio_dir) if f.endswith('.wav') or f.endswith('.WAV')]

    # Check if the list is empty
    if not new_audio_files:
        print('\n    No files found, check the filepath copied into new_audio_dir is correct')

    # track batches and files for numbering output
    batch_counter = 0
    file_counter = 0

    # total number of batches
    total_num_batches = len(new_audio_files) // batch_size

    # iterate over audio files in batches
    for i in range(0, len(new_audio_files), batch_size):
        batch_files = new_audio_files[i:i+batch_size]
        
        # write current batch to a temporary file
        with open('current_batch.txt', 'w') as f:
            for file in batch_files:
                f.write(file + '\n')

        # now run my_secondary_script.py
        subprocess.run(['python', 'child_script_v1.py', 
                        'current_batch.txt', #sys1
                        new_audio_dir, #sys2
                        output_folder, #sys3
                        model_dir, #sys4
                        str(sample_rate), #sys5
                        str(batch_counter), #sys6
                        str(total_num_batches), #sys7
                        str(file_counter)]) #sys8
        
        # Load the csv file into a DataFrame
        df = pd.read_csv(os.path.join(output_folder, 'temporary_results_table.csv'))

        # Get the number of rows in the csv (this is number of suspected bomb files written for manual checking)
        num_rows = df.shape[0]

        # Update file_counter with the number of rows
        file_counter = str(num_rows)  

        # Update batch_counter
        batch_counter += 1             
        
        # remove temporary batch file
        os.remove('current_batch.txt')

    # rename the results table csv with datetime to prevent accidental overwrites
    current_name = 'temporary_results_table.csv'
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_name = f'final_results_table_{timestamp}.csv'

    # Construct the full paths
    current_path = os.path.join(output_folder, current_name)
    new_path = os.path.join(output_folder, new_name)

    # Rename the csv file
    os.rename(current_path, new_path)


    # print findings  
    print("   Bomb detection complete on all files!\n")
    print("Found {} suspected bombs".format(num_rows)) #warning this doesnt yet capture skipped files due to corruption, or, number of all files checked
    print("The suspected bomb files and a results spreadsheet have been written to: " + output_folder +". \nYou can now check these using audacity")

if __name__ == "__main__":
    multiproc_exec()
    print("All processes have completed.")