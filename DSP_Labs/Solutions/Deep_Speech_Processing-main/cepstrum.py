import librosa
import numpy as np
from matplotlib import pyplot as plt

#Reading a wav file

def read_wav(filename):
  wav, fs = librosa.load(filename, sr = 16000)
  return wav, fs

wav, fs = read_wav('female.wav')
time = np.arange(0, wav.shape[-1])/fs

segment_len = int(40e-3 *fs)
start = 10000
voiced_segment = wav[start: start + segment_len] * np.hamming(segment_len)
len = voiced_segment.shape[-1]
voiced_segment_fft = np.log10(np.abs(np.fft.fft(voiced_segment)))


cepstrum = np.real(np.fft.ifft(voiced_segment_fft))

#High time liftering
mask_hl = np.ones(cepstrum.shape)
N = 20
mask_hl[: N] = 0
mask_hl[-N:] = 0
cepstrum_mask = mask_hl*cepstrum
high_timelifter = np.real(np.fft.fft(cepstrum_mask, 640))

# Low time liftering
mask_ll = np.abs(1-mask_hl)
cepstrum_mask = mask_ll*cepstrum
low_timelifter = np.real(np.fft.fft(cepstrum_mask, 640))
freqs = librosa.fft_frequencies(sr=fs, n_fft=640)

# get pitch
mask_p = np.zeros(cepstrum.shape)
mask_p[80:100] = 1
cepstrum_mask = mask_p*cepstrum
pitch = np.real(np.fft.fft(cepstrum_mask, 640))
# pitch = np.log10(np.abs(np.fft.ifft(pitch, 640)))

fontsize = 8
plt.figure(figsize=(5, 10))

plt.subplot(6,2,1)
plt.plot(voiced_segment)
plt.title('Voiced segment', fontsize=fontsize)
plt.subplot(6,2,3)
plt.plot(freqs, voiced_segment_fft[:len//2 + 1]) 
plt.subplot(6,2,5)
plt.plot(cepstrum[:len//2 + 1])
plt.plot(mask_hl[:len//2 + 1] * 0.25)
plt.plot(mask_ll[:len//2 + 1]* 0.25)
plt.plot(mask_p[:len//2 + 1]* 0.25)
plt.legend(['Cepstrum', 'High time liftering', 'Low time liftering', 'Pitch'], fontsize=4)
plt.ylim([-0.25, 0.6])
plt.subplot(6,2,7)
plt.plot(freqs, pitch[:len//2 + 1])
plt.subplot(6,2,9)
plt.plot(freqs, low_timelifter[:len//2 + 1])
plt.subplot(6,2,11)
plt.plot(freqs, high_timelifter[:len//2 + 1])


segment_len = int(40e-3 *fs)
start = 14600
voiced_segment = wav[start: start + segment_len] * np.hamming(segment_len)
len = voiced_segment.shape[-1]
voiced_segment_fft = np.log10(np.abs(np.fft.fft(voiced_segment)))

cepstrum = np.real(np.fft.ifft(voiced_segment_fft))

#High time liftering
mask_hl = np.ones(cepstrum.shape)
N = 20
mask_hl[: N] = 0
mask_hl[-N:] = 0
cepstrum_mask = mask_hl*cepstrum
high_timelifter = np.real(np.fft.fft(cepstrum_mask, 640))

# Low time liftering
mask_ll = np.abs(1-mask_hl)
cepstrum_mask = mask_ll*cepstrum
low_timelifter = np.real(np.fft.fft(cepstrum_mask, 640))

# get pitch
mask_p = np.zeros(cepstrum.shape)
mask_p[80:100] = 1
cepstrum_mask = mask_p*cepstrum
pitch = np.real(np.fft.fft(cepstrum_mask, 640))
# pitch = np.log10(np.abs(np.fft.ifft(pitch, 640)))


plt.subplot(6,2,2)
plt.plot(voiced_segment)
plt.title('Unvoiced segment', fontsize=fontsize)
plt.subplot(6,2,4)
plt.plot(freqs, voiced_segment_fft[:len//2 + 1]) 
plt.subplot(6,2,6)
plt.plot(cepstrum[:len//2 + 1])
plt.plot(mask_hl[:len//2 + 1]* 0.25)
plt.plot(mask_ll[:len//2 + 1]* 0.25)
plt.plot(mask_p[:len//2 + 1]* 0.25)
plt.ylim([-0.25, 0.6])
plt.subplot(6,2,8)
plt.plot(freqs, pitch[:len//2 + 1])
plt.subplot(6,2,10)
plt.plot(freqs, low_timelifter[:len//2 + 1])
plt.subplot(6,2,12)
plt.plot(freqs, high_timelifter[:len//2 + 1])

plt.tight_layout()
plt.suptitle('Cepstrum Analysis')
plt.show()


