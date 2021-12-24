import matplotlib.pyplot as plt 
import numpy as np 
from scipy import signal 
from scipy.fftpack import fft,ifft
from numpy.lib.stride_tricks import as_strided
from scipy.signal.spectral import spectrogram
import librosa
import sounddevice
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
                       
    spectrogram_matrix= np.zeros((n_fft//2+1,n_frames),dtype=np.complex_)
    
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

def plot_power_spectrogram(spec,spec_library,audio_name,win_len):
    plt.figure()
    plt.subplot(2,1,1) 
    plt.imshow(np.abs(spec)**2,origin="lower",aspect="auto") 
    plt.ylabel("Frequency [hz]")
    plt.xlabel("Time [sec]")
    plt.title("Power spectrogram,"+" "+audio_name.split(".")[0]+' '+"with window size"+' '+str(win_len)+' '+'ms.')
    plt.tight_layout(pad = 1.5)
    plt.subplot(2,1,2) 
    plt.imshow(np.abs(spec_library)**2,origin="lower",aspect="auto") 
    plt.ylabel("Frequency [hz]")
    plt.xlabel("Time [sec]")
    plt.title("Library Function Power spectrogram,"+" "+audio_name.split(".")[0]+' '+"with window size"+' '+str(win_len)+' '+'ms.')
    plt.show()

def plot_log_spectrogram(spec,spec_library,audio_name,win_len):
    plt.figure()
    plt.subplot(2,1,1) 
    plt.imshow(np.log(np.abs(spec)+0.0005),origin="lower",aspect="auto")
    plt.ylabel("Frequency [hz]")
    plt.xlabel("Time [sec]")
    plt.title("Log spectrogram,"+" "+audio_name.split(".")[0]+' '+"with window size"+' '+str(win_len)+' '+'ms.')
    plt.tight_layout(pad = 1.5)
    plt.subplot(2,1,2) 
    plt.imshow(np.log(np.abs(spec_library)+0.0005),origin="lower",aspect="auto")
    plt.ylabel("Frequency [hz]")
    plt.xlabel("Time [sec]")
    plt.title("Library Function Log spectrogram,"+" "+audio_name.split(".")[0]+' '+"with window size"+' '+str(win_len)+' '+'ms.')
    
    plt.show()

# Problem 2
# Now run your implementation in Problem 1 with different window sizes
# for different types of signals,

signal_files = ["audio1.wav","audio2.wav","summedup.wav"]
i=0
for signal_file in signal_files:
    audio, fs = soundfile.read(signal_file)
    #use 16ms, 32 ms, 64 ms, and 128 ms.
    window_sizes = [16,32,64,128]
    window_size=[]
    for i in range(len(window_sizes)):
        window_size.append(int(window_sizes[i]*0.001*(fs)))
    k=0
    for window in window_size:
        n_fft = window
        hop_size= window//2

        spec= STFT(audio, n_fft,window, hop_size)

        #Calculate spectrogram with a library function
        spec_library = librosa.core.stft(audio,n_fft=window, win_length=window,hop_length=hop_size)
       
        # plot_power_spectrogram( spec,spec_library,signal_file,window_sizes[k])
        # plot_log_spectrogram( spec,spec_library,signal_file,window_sizes[k])
        # compare the power and log spectrograms for different signals and window sizes 
        plt.figure(i)
        plt.subplot(4,1,k+1) 
        plt.imshow(np.log(np.abs(spec)+0.0005),origin="lower",aspect="auto")
        plt.ylabel("Frequency [hz]")
        plt.xlabel("Time [sec]")
        plt.title("Log spectrogram,"+" "+signal_file.split(".")[0]+' '+"with window size"+' '+str(window_sizes[k])+' '+'ms.')
        plt.tight_layout(pad = 0.3)

        k+=1
    i+=1
    plt.show()


