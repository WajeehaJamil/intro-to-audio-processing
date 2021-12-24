import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import scipy.io.wavfile
import scipy.fftpack
import sounddevice
from scipy.io.wavfile import read

# read audio sample 
def readAndPlayAudio(filename):
    input_signal = read(filename)
    audio = input_signal[1]
    fs=input_signal[0]
    sounddevice.play(audio,fs)
    return audio,fs

#  Plot entire signal
def plotSignal(title,audio):
    plt.plot(audio)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.title(title+"File Entire Signal")
    plt.show()
    return 

#  Plot signal between 0.5 and 1 s.
def plotSignalSegment(title,audio,fs):
    plt.plot(audio[int(0.5*fs):int(1*fs)])
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.title(title+"File Signal between 0.5 and 1 s")
    plt.show()
    return


def computingDFT(start,audio,increment,end):
    # plotting the first segment and corresponding magnitude DFT
    plt.subplot(2,1,1) 
    plt.plot(audio[int(start*fs):int((start+increment)*fs)])
    plt.title("Plotting the first segment")
    plt.tight_layout(pad = 1.5)
    DFT=np.abs(scipy.fftpack.fft(audio[int(start*fs):int((start+increment)*fs)], n=512))
    plt.subplot(2,1,2)
    plt.plot(DFT)
    plt.title(" Plotting the corresponding magnitude DFT of first segment")
    plt.show()
    
    #loop basically computes magnitude DFT for each of the 100 ms segments of the
    #audio
    for x in np.arange(start, end, increment):
        DFT=np.abs(scipy.fftpack.fft(audio[int(start*fs):int((start+increment)*fs)], n=512))
        start+=increment
    return

# read audio sample 1
audio,fs = readAndPlayAudio("audio1.wav")
plotSignal("Audio 1",audio)
plotSignalSegment("Audio 1",audio,fs)
#Read the next (from 1sec onwards) 100 ms of the signal.
increment=0.1
start_index=1
end_index=len(audio)/fs
computingDFT(start_index,audio,increment,end_index)

# read audio sample 2
audio,fs = readAndPlayAudio("audio2.wav")
plotSignal("Audio 2",audio)
plotSignalSegment("Audio 2",audio,fs)
#Read the next (from 1sec onwards) 100 ms of the signal.
increment=0.1
start_index=1
end_index=len(audio)/fs
computingDFT(start_index,audio,increment,end_index)



