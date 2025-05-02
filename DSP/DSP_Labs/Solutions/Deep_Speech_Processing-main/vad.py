import numpy as np
import librosa
from matplotlib import pyplot as plt
from scipy.fft import fft
from scipy.stats import entropy

def spectral_entropy(signal, fft_size=None, eps=1e-10):
    if fft_size is None:
        fft_size = len(signal)

    spectrum = np.abs(fft(signal, n=fft_size))[:fft_size // 2]  # Keep only positive freqs

    psd = spectrum**2
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0
    prob_dist = psd / psd_sum

    return entropy(prob_dist + eps, base=2) 

def zero_crossing_rate(signal):
    return np.mean(np.abs(np.diff(np.sign(signal)))) / 2

def frame_signal(signal, window_length, hop_length, window_function=np.hanning):
    pad_size = window_length // 2
    signal = np.pad(signal, (pad_size, pad_size), mode='constant') 
    num_frames = 1 + (len(signal) - window_length) // hop_length 
    frames = np.zeros((num_frames, window_length)) 

    energy = np.zeros(num_frames)
    zcr = np.zeros(num_frames)
    entropy = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_length
        end = start + window_length
        frames[i] = signal[start:end] * window_function(window_length)
        energy[i] = np.mean(frames[i]**2)
        zcr[i] = zero_crossing_rate(frames[i])
        entropy[i] = spectral_entropy(frames[i])
    time = np.arange(num_frames) * hop_length
    return frames, time, energy, zcr, entropy


def read_wav(filename):
  wav, fs = librosa.load(filename, sr = 16000)
  return wav, fs

wav, fs = read_wav('female.wav')
time = np.arange(0, wav.shape[-1])/fs
hop_length = int(fs*2e-3)

ste_win_len = 40e-3
lte_win_len = 200e-3

window_length = int(fs*ste_win_len)
_ ,_,st_frames_energy, st_zcr, st_entropy  = frame_signal(wav, window_length, hop_length) 

window_length = int(fs*lte_win_len)
_ , _, lt_frames_energy, lt_zcr, lt_entropy = frame_signal(wav, window_length, hop_length)

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(time, wav)
plt.margins(x=0)
plt.subplot(3,1,2)
plt.plot(st_frames_energy)
plt.plot(lt_frames_energy)
plt.margins(x=0)
plt.legend(['ste','lte'])
plt.subplot(3,1,3)
plt.plot(st_frames_energy> lt_frames_energy)
plt.title('Energy based VAD')
plt.margins(x=0)
plt.tight_layout()

ste_win_len = 15e-3
lte_win_len = 250e-3

window_length = int(fs*ste_win_len)
_ ,_,st_frames_energy, st_zcr, st_entropy  = frame_signal(wav, window_length, hop_length) 

window_length = int(fs*lte_win_len)
_ , _, lt_frames_energy, lt_zcr, lt_entropy = frame_signal(wav, window_length, hop_length)

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(time, wav)
plt.margins(x=0)
plt.subplot(3,1,2)
plt.plot(st_zcr)
plt.plot(lt_zcr)
plt.margins(x=0)
plt.legend(['zcr ste','zcr lte'])
plt.subplot(3,1,3)
plt.plot(st_zcr < lt_zcr)
plt.title('ZCR based VAD')
plt.margins(x=0)
plt.tight_layout()

ste_win_len = 10e-3
lte_win_len = 10e-3

window_length = int(fs*ste_win_len)
_ ,_,st_frames_energy, st_zcr, st_entropy  = frame_signal(wav, window_length, hop_length) 

window_length = int(fs*lte_win_len)
_ , _, lt_frames_energy, lt_zcr, lt_entropy = frame_signal(wav, window_length, hop_length)

entropy_thresh = 4

plt.figure(3)
plt.subplot(3,1,1)
plt.plot(time, wav)
plt.margins(x=0)
plt.subplot(3,1,2)
plt.plot(st_entropy)
plt.plot(np.ones(st_entropy.shape[-1])*entropy_thresh)
plt.margins(x=0)
plt.legend(['entropy ste'])
plt.subplot(3,1,3)
plt.plot(st_entropy < entropy_thresh)
plt.title('Entropy based VAD')
plt.margins(x=0)
plt.tight_layout()
plt.show()