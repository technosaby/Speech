import numpy as np
import librosa
from matplotlib import pyplot as plt
import scipy

wav, fs = librosa.load('male.wav', sr=16000)
wav = wav[int(fs*6.9):int(fs*8.6)]

time = np.arange(0, wav.shape[-1])/fs
time = time[:wav.shape[-1]]
win_len_sec = 60e-3
win_len = int(win_len_sec*fs)
nfft =  win_len*2
noverlap = int((win_len_sec/2)*fs)
window = np.hamming(win_len)
# window = np.ones(win_len)

f, t, spec = scipy.signal.spectrogram(wav, fs=fs, window=window, nperseg=win_len, noverlap=noverlap, nfft=nfft, \
                         detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')

plt.subplot(2,1,1)
plt.margins(x=0, y=0)
plt.plot(time,wav)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('Time domain waveform')
plt.subplot(2,1,2)
plt.imshow(np.log10(spec), aspect='auto', origin='lower', cmap='viridis', extent=[t.min(), t.max(), f.min(), f.max()])
plt.xlabel('time')
plt.ylabel('Frequency')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()