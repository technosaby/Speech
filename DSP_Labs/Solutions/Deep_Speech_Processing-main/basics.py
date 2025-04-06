import numpy as np
import librosa
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import correlation_lags

def autocorrelation(signal):
    n = signal.shape[-1]
    lags = correlation_lags(n, n, mode='full')
    mean = np.mean(signal)
    variance = np.var(signal)
    normalized_signal = (signal - mean) / np.sqrt(variance * n)
    result = np.correlate(normalized_signal, normalized_signal, mode='full')
    # result[result.size // 2:]
    return result, lags

def read_wav(filename):
  wav, fs = librosa.load(filename, sr = 16000)
  return wav, fs

wav, fs = read_wav('female.wav')
time = np.arange(0, wav.shape[-1])/fs

# Voiced segment
segment_len = int(30e-3 *fs)
start = 10000
voiced_segment = wav[start: start + segment_len] * np.hamming(segment_len)
len = voiced_segment.shape[-1]
len = 3*len
voiced_segment_fft = np.abs(np.fft.fft(voiced_segment, len))[:len//2 + 1]

freqs = librosa.fft_frequencies(sr=fs, n_fft=len)
v_autocorr, lags = autocorrelation(voiced_segment)

#Unvoiced segment
segment_len = int(30e-3 *fs)
start = 14600
unvoiced_segment = wav[start: start + segment_len] * np.hamming(segment_len)
len = unvoiced_segment.shape[-1]
len = 3*len
unvoiced_segment_fft = np.abs(np.fft.fft(unvoiced_segment, len))[:len//2 + 1]
uv_autocorr, lags = autocorrelation(unvoiced_segment)

#plotting
time_domain = [voiced_segment, unvoiced_segment]
mag_fft = [voiced_segment_fft, unvoiced_segment_fft]
log_mag_fft = [np.log10(voiced_segment_fft), np.log10(unvoiced_segment_fft)]
autocorr = [v_autocorr, uv_autocorr]

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(5, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time, wav)
ax1.set_xlabel('time[sec]')
ax1.set_title('Time-Domain Waveform')
ax1.margins(x=0, y=0)

title = ['voiced', 'unvoiced']
for j in range(2):
   ax = fig.add_subplot(gs[1, j])
   ax.plot(time_domain[j])
   ax.set_title(title[j])
   ax.set_xlabel('time[samples]')
   ax.margins(x=0, y=0)

for j in range(2):
   ax = fig.add_subplot(gs[2, j])
   ax.plot(freqs, mag_fft[j])
   ax.set_title('Magnitude spectrum')
   ax.set_xlabel('freq[Hz]')
   ax.margins(x=0, y=0)

for j in range(2):
   ax = fig.add_subplot(gs[3, j])
   ax.plot(freqs, log_mag_fft[j])
   ax.set_title('Log-Magnitude spectrum')
   ax.set_xlabel('freq[Hz]')
   ax.margins(x=0, y=0)

for j in range(2):
   ax = fig.add_subplot(gs[4, j])
   ax.plot(lags, autocorr[j])
   ax.set_title('Autocorrelation')
   ax.set_xlabel('Lag[samples]')
   ax.margins(x=0, y=0)

plt.tight_layout()
plt.show()