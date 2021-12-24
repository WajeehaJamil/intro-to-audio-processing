import matplotlib.pyplot as plt 
import numpy as np 
from scipy import signal 
from scipy.fftpack import fft,ifft
from numpy.lib.stride_tricks import as_strided
import soundfile

# function to compute short-time fourier transform
def STFT(s,n_fft,winsize,hopsize):
    # introducing hamming window
    win_funct = signal.hamming(winsize,sym=False)
    window_analysis = np.sqrt(win_funct)

    ## (b)
    # introducing 50 % overlap between consecutive signal frames   
    n_frames=int((len(s)-winsize)/hopsize)+1
    y_frames=as_strided(s,(winsize,n_frames),(s.itemsize,hopsize*s.itemsize))
                            #shape             #strides (step) 
    ## (a)
    #  plotting magnitude DFT for windowing and non windowing first segment 
    first_segment= y_frames[:,1]
    plt.subplot(4,1,1) 
    plt.plot(first_segment)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.title("Plotting the first segment")
    DFT=np.abs(fft(first_segment))
    plt.subplot(4,1,2)
    plt.plot(DFT)
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency")
    plt.title(" Plotting the corresponding magnitude DFT of first segment")
  
    windowed_segment= first_segment*window_analysis
    plt.subplot(4,1,3) 
    plt.plot(windowed_segment)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.title("Plotting the windowed first segment")
    plt.tight_layout(pad = 1.5)
    DFT_windowed=np.abs(fft(windowed_segment))
    plt.subplot(4,1,4)
    plt.plot(DFT_windowed)
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency")
    plt.title(" Plotting the corresponding magnitude DFT of windowed first segment")
    plt.show()                      
    spectrogram_matrix = np.zeros((n_fft//2+1,n_frames),dtype=np.complex_)
    
    ## This loop does two things:
    # 1. Multiply each signal frame with a windowing function (use Hamming)
    # 2. Collect the power spectrum into a matrix such that each
    #    column of the matrix contains the power spectrum of the nth frame, n =0, 1, 2, ....
    for i in np.arange(n_frames):
        a=y_frames[:,i]
        window_frame=a*window_analysis

        spectrum=fft(window_frame)
        spectrum=spectrum[:n_fft//2+1]

        spectrogram_matrix[:,i]=spectrum
    return spectrogram_matrix

def plot_power_spectrogram(spec,audio_name):
    plt.imshow(np.abs(spec)**2,origin="lower",aspect="auto") 
    plt.ylabel("Frequency [hz]")
    plt.xlabel("Time [sec]")
    plt.title("Power spectrogram,"+" "+audio_name.split(".")[0])

def plot_log_spectrogram(spec,audio_name):
    plt.imshow(np.log(np.abs(spec)+0.0005),origin="lower",aspect="auto")
    plt.ylabel("Frequency [hz]")
    plt.xlabel("Time [sec]")
    plt.title("Log spectrogram,"+" "+audio_name.split(".")[0])

# read audio sample 1
audio, fs = soundfile.read("audio1.wav")

win_size = fs
n_fft = win_size
hop_size= win_size//2
spect= STFT(audio,n_fft,win_size,hop_size)
plt.figure()
plt.subplot(2,1,1)
plot_power_spectrogram(spect,"audio1.wav")
plt.tight_layout(pad = 1.5)
plt.subplot(2,1,2)
plot_log_spectrogram(spect,"audio1.wav")
plt.show()
