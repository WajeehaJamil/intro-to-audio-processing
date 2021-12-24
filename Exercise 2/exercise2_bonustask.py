import matplotlib.pyplot as plt 
import numpy as np 
from scipy import signal 
from scipy.fftpack import fft,ifft
from numpy.lib.stride_tricks import as_strided
import soundfile

def plotSignal(title,audio,fs):
    plt.plot(audio[int(0.5*fs):int(1*fs)])
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.title(title)
    return 

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

# function to compute inverse DFT (IDFT) in a loop with the same window size, overlap and
# window type as used in analysis
def ISTFT(spectrogram_long,s,winsize,hopsize):
    
    win_funct = signal.hamming(winsize,sym=False)
    window_synthesis = np.sqrt(win_funct)
    
    n_frames=spectrogram_long.shape[1]
    result=np.zeros((s.shape[0]))
    
    for i in np.arange(n_frames):
        a=spectrogram_long[:,i]
        b=np.conjugate(spectrogram_long[:,i][-2:0:-1])
        c=np.concatenate((a,b))
        out_long=ifft(c).real
        out_long_wnd=window_synthesis*out_long
        result[i*hopsize:i*hopsize+2*hopsize]=result[i*hopsize:i*hopsize+2*hopsize]+out_long_wnd

    return result 
    

audio,fs  = soundfile.read('audio1.wav')
    
window_length=int(64*0.001*fs)
n_fft = window_length
hop_size = window_length//2
spec=STFT(audio,n_fft,window_length,hop_size)
audio_=ISTFT(spec,audio,window_length,hop_size)

# Plot and compare your sinusoidal signal with the reconstructed signal
plt.figure()
plt.subplot(2,1,1) 
plotSignal("Original Signal",audio,fs)
plt.tight_layout(pad = 1.5)
plt.subplot(2,1,2) 
plotSignal("Reconstructed Signal",audio_,fs)
plt.show()
   
