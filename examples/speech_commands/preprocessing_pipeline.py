'''
Pre-processing pipeline
-Creates a training set and performs following preprocessing steps:
-1. Re-sample the data from 16KHz to 8kHz
-2. Segment files based on voice activity detection (VAD)
-Save the result of each pre-processing step in a different directory
Robert Brais
'''

import create_annotation_subset as cas
import resample
import vad

def main():
    training_set = cas.get_set('training')
    dest_directory = 'C:\\Users\\rober\\Documents\\Projects\\annotation_dataset\\'
    cas.save_set(training_set,dest_directory)
    #resample
    #vad
