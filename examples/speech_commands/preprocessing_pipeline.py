'''
Pre-processing pipeline
-Creates a training set and performs following preprocessing steps:
-1. Re-sample the data from 16KHz to 8kHz
-2. Segment files based on voice activity detection (VAD)
-Save the result of each pre-processing step in a different directory
Robert Brais
'''
from scipy.io import wavfile
from scipy import signal
from os import path
import numpy as np

import create_annotation_subset as cas
import resample
import vad

def main():
    training_set = cas.get_set('testing')
    raw_directory = 'C:\\Users\\rober\\Documents\\Projects\\resample_dataset\\'
    cas.save_set(training_set,raw_directory)

    #resample
    resample_directory = 'C:\\Users\\rober\\Documents\\Projects\\resample_dataset\\'
    new_sample_rate = 8000
    for training_sample in training_set:
        sample_rate, samples = wavfile.read(training_sample['file'])
        resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        head,id = path.split(training_sample['file'])
        partial_path = path.join(resample_directory, training_sample['label'])
        full_path = path.join(partial_path, id)
        resampled = np.asarray(resampled, dtype=np.int16)
        wavfile.write(full_path,new_sample_rate,resampled)
        #debug plots
        # resample.comparison_plot(samples, sample_rate, resampled, new_sample_rate)

    #vad
if __name__ == '__main__':
    main()