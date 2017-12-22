'''
resample.py
-Resamples 16kHz wav to 8kHz by removing highest frequencies
https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
Robert Brais
'''

import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile

import matplotlib.pyplot as plt

def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    vals = 2.0/N * np.abs(yf[0:N//2])  # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    return xf, vals

def comparison_plot(samples,sample_rate,resampled,new_sample_rate):
    xf, vals = custom_fft(samples, sample_rate)
    plt.figure(figsize=(12, 4))
    plt.title('FFT of recording sampled with ' + str(sample_rate) + ' Hz')
    plt.plot(xf, vals)
    plt.xlabel('Frequency')
    plt.grid()
    plt.show()

    xf, vals = custom_fft(resampled, new_sample_rate)
    plt.figure(figsize=(12, 4))
    plt.title('FFT of recording sampled with ' + str(new_sample_rate) + ' Hz')
    plt.plot(xf, vals)
    plt.xlabel('Frequency')
    plt.grid()
    plt.show()

def main():
    filename = 'C:\\Users\\rober\\Documents\\Projects\\speech_dataset\\happy\\0b09edd3_nohash_0.wav'
    new_sample_rate = 8000

    #resample
    sample_rate, samples = wavfile.read(filename)
    resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))

    #plot
    comparison_plot(samples, sample_rate, resampled, new_sample_rate)

if __name__ == '__main__':
    main()