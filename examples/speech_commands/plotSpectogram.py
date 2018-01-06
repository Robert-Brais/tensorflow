'''
Plot Spectogram
-Plot a visualization of the wav amplitude over time and spectogram
Based on :
https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
Robert Brais
'''
# Math
import numpy as np
from scipy import signal
from scipy.io import wavfile

# Visualization
import matplotlib.pyplot as plt

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def main(audio_path,filename):
    sample_rate, samples = wavfile.read(str(audio_path) + filename)
    freqs, times, spectrogram = log_specgram(samples, sample_rate)

    # Plot
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave of ' + filename)
    ax1.set_ylabel('Amplitude')
    # ax1.plot(np.linspace(0, sample_rate / len(samples), sample_rate), samples)
    ax1.plot(np.linspace(0,  len(samples) / sample_rate, len(samples)), samples)

    ax2 = fig.add_subplot(212)
    ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    ax2.set_yticks(freqs[::16])
    ax2.set_xticks(times[::16])
    ax2.set_title('Spectrogram of ' + filename)
    ax2.set_ylabel('Freqs in Hz')
    ax2.set_xlabel('Seconds')

    plt.show()

if __name__ == "__main__":
    # set the path
    # train_audio_path = 'C:\\Users\\rober\\Documents\\Projects\\resample_dataset'
    train_audio_path = 'C:\\Users\\rober\\Documents\\Projects\\vad_dataset'
    # filename = '\\down\\bdee441c_nohash_3.wav'
    # filename = '\\yes\\2296b1af_nohash_0_00.wav'
    filename = '\\chunk-00.wav'
    main(train_audio_path,filename)


