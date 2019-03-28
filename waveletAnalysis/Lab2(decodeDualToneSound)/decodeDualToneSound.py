#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

#here we find quantity peaks
fs,signal=wav.read("e:\My\s2.wav ")
c = (signal)
plt.plot(abs(c[:int(len(c))]),'r') 
plt.show()


# In[ ]:


import wave
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

n=22 # n = quantity peaks   14,22,21

def getPicks(c):
    a =[]
    for i in range(len(c)):
        if c[i]>100:
            a.append(i*n/nframes*fs)
    return a


nframes = wave.open("e:\My\s2.wav ", mode="r").getnframes()
fs,signal=wav.read("e:\My\s2.wav ")
signal1 = signal/32767 #2**15-1
t = [i/fs for i in range(len(signal1))]
f = [i for i in range(len(signal1))]
for i in range(n):
    c = fft(signal1[int(len(signal1)/n*(i)):int(len(signal1)/n*(i+1))])
    x = [i*n/nframes*fs for i in range(int(len(c)/2))]
    plt.plot(x, abs(c[:int(len(c)/2)]),'r') 
    plt.show()
    print(getPicks(abs(c[:int(len(c)/2)])))


# In[ ]:


#Res for s1 0123456789ABCD
#Res for s2 A1B2C3D4#A5B6C7D8#A9B0
#Res for s3 ABBCCCDDDD00000111111



