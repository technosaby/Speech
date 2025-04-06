import numpy as np
from matplotlib import pyplot as plt

def aliased_frequency(f, fs):
    fn = fs / 2
    f_alias = abs(f % fs)
    if f_alias > fn:
        f_alias = abs(fs - f_alias)
    return f_alias

def get_sin(f, fs, duration, phi=0):
    t = np.arange(0, duration, 1/fs)
    sig = np.sin(2*np.pi*f*t + phi)
    return sig, t

f_alias = []

fs = 100
duration = 2

f1 = 5
sig1, t = get_sin(f1, fs, duration)

if 2*f1 > fs:
    f_alias.append(aliased_frequency(f1, fs))
    
f2 = 26
sig2, t = get_sin(f2, fs, duration)
if 2*f2 > fs:
    f_alias.append(aliased_frequency(f2, fs))

sig = sig1 + sig2
sig_fft = np.abs(np.fft.fft(sig, 2048))
freq = np.arange(0, fs, fs/sig_fft.shape[-1])
len = sig_fft.shape[-1]


# plotting
aliased_freq = np.zeros(sig_fft.shape)
for i in f_alias:
    # index = freq == i
    index = np.argmin(np.abs(freq-i))
    aliased_freq[index] = np.max(np.abs(sig_fft))


plt.subplot(2,1,1)
plt.plot(t, sig)
plt.title(f'Time domain, Freq = [{f1}, {f2} Hz]')
plt.xlabel('time')
plt.ylabel('ampplitude')
plt.subplot(2,1,2)
plt.plot(freq[:len//2 + 1], sig_fft[:len//2 + 1])
plt.stem(freq[:len//2 + 1], aliased_freq[:len//2 + 1], 'r')
plt.xlabel('Freq. [HZ]')
plt.ylabel('Magnitude')
plt.title('Freq. magnitude spectrum')
plt.legend(['','Aliased frequency'])
plt.tight_layout()
plt.show()