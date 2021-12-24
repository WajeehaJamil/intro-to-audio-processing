import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import scipy.io.wavfile
import scipy.fftpack
import scipy.signal
import sounddevice
from scipy.io.wavfile import read

# create sinusoids
fs = 8000
t = 3

def create_sinusoids(freq, amp, phase):
    t_seq=np.linspace(0,t,fs*3)
    y = amp*np.sin(2*np.pi*freq*t_seq+phase)
    
    plt.figure(1)
    plt.subplot(5, 1, amp)
    plt.plot(t_seq[:300],y[:300])
    plt.title(str(freq)+' Hz Sinusoid')
    plt.tight_layout(pad = 0.3)
    sounddevice.play(y,fs)
    return y

# Create 4 sinusoids
# plot and play them
y1 = create_sinusoids(100, 1, np.pi)
y2 = create_sinusoids(500, 2, np.pi/2)
y3 = create_sinusoids(1500, 3, 2*np.pi)
y4 = create_sinusoids(2500, 4, 3*np.pi)


#c) Add them up to x(t). Plot and play x(t). Write the signal to a wav file.

x=y1+y2+y3+y4
t_seq=np.linspace(0,3,fs*3)
plt.subplot(5, 1, 5)
plt.plot(t_seq[:300],x[:300])
plt.title('SUMMED UP wave of 4 sinusoids')
plt.show()
scipy.io.wavfile.write('summedup.wav',8000,y1)
sounddevice.play(x,fs)

#d) Apply DFT. Plot magnitude DFT.
DFT=np.abs(scipy.fftpack.fft(x, n=512))
plt.plot(DFT)
plt.title("Magnitude DFT of original summedup signal")
plt.show()

#BONUS TASK
#In problem1 downsample the sum of sinusoids x(t) by a factor of 2
A = 4
fs=8000
newsampling_frequency=int(fs/2)
newnumberof_samples = int(len(x)/2)

downsample = scipy.signal.resample(x,newnumberof_samples)
scipy.io.wavfile.write("downsampled.wav", newsampling_frequency, downsample)

#Plot the dawnsampled signal
t=np.linspace(0,3,fs*3)
plt.plot(t[:300], downsample[:300])
plt.ylabel("Amplitude")
plt.xlabel("Time") 
plt.title("Downsampled Wav audio")
plt.show()


#Plot and compare magnitude DFT of original and downsampled signal.
DFTdownsampled=np.abs(scipy.fftpack.fft(downsample, n=512))
plt.subplot(2, 1, 1)
plt.plot(DFT)
plt.title("Magnitude DFT of original summedup signal")
plt.tight_layout(pad = 1.5)
plt.subplot(2, 1, 2)
plt.plot(DFTdownsampled)
plt.title("Magnitude DFT of downsampled summedup signal")
plt.show()






