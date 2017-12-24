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
from vad import frame_generator
from vad import vad_collector
from vad import write_wave
import webrtcvad

def main():
    dest_directory = 'C:\\Users\\rober\\Documents\\Projects\\vad_dataset\\'

    training_set = cas.get_set('testing')
    # cas.save_set(training_set,dest_directory)

    #resample
    dest_directory = 'C:\\Users\\rober\\Documents\\Projects\\vad_dataset\\'
    new_sample_rate = 8000

    #vad
    vad_agressiveness = 0 #0-3
    vad = webrtcvad.Vad(vad_agressiveness)

    #preprocessing loop
    for training_sample in training_set:

        #resample
        sample_rate, samples = wavfile.read(training_sample['file'])
        resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        head,id = path.split(training_sample['file'])
        partial_path = path.join(dest_directory, training_sample['label'])
        full_path = path.join(partial_path, id)
        resampled = np.asarray(resampled, dtype=np.int16)
        # wavfile.write(full_path,new_sample_rate,resampled)
        #debug plots
        # resample.comparison_plot(samples, sample_rate, resampled, new_sample_rate)

        #vad
        vad = webrtcvad.Vad(vad_agressiveness)
        frames = frame_generator(10, resampled, new_sample_rate)
        frames = list(frames)
        segments = vad_collector(new_sample_rate, 10, 300, vad, frames)
        for i, segment in enumerate(segments):
            chunk_name = '_%002d.wav' % (i,)
            full_path = full_path.replace('.wav', chunk_name)
            write_wave(full_path, segment, new_sample_rate)

if __name__ == '__main__':
    main()