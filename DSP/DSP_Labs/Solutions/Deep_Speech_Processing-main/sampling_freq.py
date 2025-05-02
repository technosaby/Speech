import numpy as np
from matplotlib import pyplot as plt

def get_sin(f, fs, duration, phi=0):
    t = np.arange(0, duration, 1/fs)
    sig = np.sin(2*np.pi*f*t + phi)
    return sig, t

f_alias = []

fs = 30
duration = 0.5

f = 5
sig_analog, t_analog = get_sin(f, 100, duration)

sig, t = get_sin(f, fs, duration)
fft_points = int(sig.shape[-1]*16)

sig_fft = np.fft.fft(sig, fft_points)
sig_ifft = np.fft.ifft(sig_fft)[:sig.shape[-1]]
freqs = np.arange(0, fs, fs/sig_fft.shape[-1])[:sig_fft.shape[-1]]

font_size = 5

plt.subplot(4,2,1)
plt.plot(t_analog, sig_analog)
plt.stem(t, sig, 'r')
plt.legend(['analog', 'discrete'],  fontsize=font_size)
plt.title(f'No. of samples, N =  {sig.shape[-1]}', fontsize=font_size)
plt.xlabel('time [sec]', fontsize=font_size)
plt.ylabel('Amplitude', fontsize=font_size)
plt.subplot(4,2,2)
plt.xlabel('freq [Hz]', fontsize=font_size)
plt.plot(freqs, np.abs(sig_fft))
plt.title(f'Mag. spectrum is cont.', fontsize=font_size)
plt.xlabel('Freq. [Hz]', fontsize=font_size)
plt.ylabel('Magnitude', fontsize=font_size)
plt.tight_layout()


plt.subplot(4,2,3)
plt.plot(freqs, np.abs(sig_fft))
plt.stem(freqs, np.abs(sig_fft))
plt.xlabel('Freq. [Hz]', fontsize=font_size)
plt.ylabel('Magnitude', fontsize=font_size)
plt.title(f'Sampled magnitude spectrum, FFT points = {fft_points}.', fontsize=font_size)
plt.subplot(4,2,4)
plt.stem(t, sig, 'b')
plt.stem(t, np.real(sig_ifft), 'r')
plt.xlabel('time [sec]', fontsize=font_size)
plt.ylabel('Amplitude', fontsize=font_size)
plt.legend(['original', 'Reconstructed'],  fontsize=font_size)
plt.title(f'Reconstructed signal, FFT points [240] > No. of samples [15] ', fontsize=font_size)
plt.tight_layout()

sig_fft_downsample = sig_fft[::16]
sig_ifft_downsample = np.fft.ifft(sig_fft_downsample)
mask = np.zeros(sig_fft.shape[-1])
mask[::16]=1

plt.subplot(4,2,5)
plt.plot(freqs, np.abs(sig_fft))
plt.stem(freqs, np.abs(mask*sig_fft))
plt.xlabel('Freq. [Hz]', fontsize=font_size)
plt.ylabel('Magnitude', fontsize=font_size)
plt.title(f'Sampled magnitude spectrum, FFT points = 15.', fontsize=font_size)
plt.subplot(4,2,6)
plt.stem(t, sig, 'b')
plt.stem(t, np.real(sig_ifft_downsample), 'r')
plt.xlabel('time [sec]', fontsize=font_size)
plt.ylabel('Amplitude', fontsize=font_size)
plt.legend(['original', 'Reconstructed'],  fontsize=font_size)
plt.title(f'Reconstructed signal, FFT points [15] = No. of samples [15] ', fontsize=font_size)
plt.tight_layout()

sig_fft_downsample = sig_fft_downsample[::2]
sig_ifft_downsample = np.fft.ifft(sig_fft_downsample)
sig_ifft = np.zeros(sig.shape[-1])
sig_ifft[:sig_ifft_downsample.shape[-1]] = sig_ifft_downsample
mask = np.zeros(sig_fft.shape[-1])
mask[::32] = 1

plt.subplot(4,2,7)
plt.plot(freqs, np.abs(sig_fft))
plt.stem(freqs, np.abs(mask*sig_fft))
plt.xlabel('Freq. [Hz]', fontsize=font_size)
plt.ylabel('Magnitude', fontsize=font_size)
plt.title(f'Sampled magnitude spectrum, FFT points = 8.', fontsize=font_size)
plt.subplot(4,2,8)
plt.stem(t, sig, 'b')
plt.stem(t, np.real(sig_ifft), 'r')
plt.xlabel('time [sec]', fontsize=font_size)
plt.ylabel('Amplitude', fontsize=font_size)
plt.legend(['original', 'Reconstructed'],  fontsize=font_size)
plt.title(f'Reconstructed signal, FFT points [8] < No. of samples [15] ', fontsize=font_size)

plt.suptitle('Frequency Sampling')
plt.tight_layout()

plt.show()