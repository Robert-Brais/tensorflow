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
from vad import read_wave
import webrtcvad
import plotSpectogram

def main():
    dest_directory = 'C:\\Users\\rober\\Documents\\Projects\\vad_dataset\\'

    training_set = cas.get_set('testing')
    # cas.save_set(training_set,dest_directory)

    #resample
    dest_directory = 'C:\\Users\\rober\\Documents\\Projects\\vad_dataset\\'
    new_sample_rate = 8000

    #vad
    src_directory = 'C:\\Users\\rober\\Documents\\Projects\\resample_dataset\\'
    vad_agressiveness = 1 #0-3

    #preprocessing loop
    for training_sample in training_set:

        #path to result file
        head,id = path.split(training_sample['file'])
        full_id = path.join(training_sample['label'],id)
        dest_path = path.join(dest_directory, full_id)

        #resample
        # sample_rate, samples = wavfile.read(training_sample['file'])
        # resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        # resampled = np.asarray(resampled, dtype=np.int16)
        # wavfile.write(dest_path,new_sample_rate,resampled)
        #debug plots
        # resample.comparison_plot(samples, sample_rate, resampled, new_sample_rate)

        #vad
        audio, sample_rate = read_wave(path.join(src_directory,full_id))
        vad = webrtcvad.Vad(vad_agressiveness)
        frames = frame_generator(10, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, 10, 300, vad, frames)
        for i, segment in enumerate(segments):
            chunk_name = '_%002d.wav' % (i,)
            dest_path = dest_path.replace('.wav', chunk_name)
            write_wave(dest_path, segment, new_sample_rate)
            #debug plot
            # plotSpectogram.main(dest_directory,full_id.replace('.wav', chunk_name))

if __name__ == '__main__':
    main()