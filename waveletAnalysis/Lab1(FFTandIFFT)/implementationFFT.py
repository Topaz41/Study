#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math as math
import random
import time
from scipy import linalg, sparse
import warnings
warnings.filterwarnings('ignore')

matrix_type = 'coo'

def I_n(n):
    return sparse.identity(n, dtype=float, format='coo')

def omega_n(n):
    return np.exp(((-(complex(0, 2))*np.pi)/(n)))

def omega_n1(n):
    return np.exp((((complex(0, 2))*np.pi)/(n)))

def Omega_n(n):
    o = omega_n(2*n)
    a = [o**i for i in range(n)]
    return sparse.diags(a)

def B_n(n):
    m = int(n/2)
    I = I_n(m)
    O = Omega_n(m)
    return sparse.bmat([[I, O], [I, -O]]).toarray()

def reverse_bit(num, osn):
    result = 0;
    while osn:
        osn-=1
        result += result + (num % 2)
        num >>= 1
    return result

def swapRow(A, n):
    a = np.zeros(shape=(2**n), dtype=complex)
    for i in range(2**n):
        a[i] = (A[reverse_bit(i,n)])
    return a

def myfft(N,x):
    x = ((swapRow(x,N))).transpose()
    for i in range(1, N+1):
        s = 2**i
        ss = int(s/2)
        o = omega_n(s)
        a = [omega_n(s)**i for i in range(ss)]
        for j in range(2**(N-i)):
            cur = x[s*j:s*(j + 1)].copy()
            for k in range(ss):
                x[s*j + k] = cur[k] + cur[k + ss] * a[k]
            for k in range(ss):
                x[s*j + k + ss] = cur[k] - cur[k + ss] * a[k]
    return x

def Difft(x, N):
    x = ((swapRow(x,N))).transpose()
    for i in range(1, N+1):
        s = 2**i
        ss = int(s/2)
        a = [omega_n1(s)**i for i in range(ss)]
        for j in range(2**(N-i)):
            cur = x[s*j:s*(j + 1)].copy()
            for k in range(ss):
                x[s*j + k] = cur[k] + cur[k + ss] * a[k]
            for k in range(ss):
                x[s*j + k + ss] = cur[k] - cur[k + ss] * a[k]
    return x/2**N


# In[2]:


from scipy.fftpack import fft, ifft 
n = 3
x =[] 
for i in range(2**n): 
    x.append(math.ceil(random.random()*10)) 
print("x",x) 

Myfft = myfft(n,x)

print("after Myfft",Myfft)

Myifft1 = Difft(Myfft,n)

_fft = fft(x) 

print("fft",_fft)

print("after MyIfft",Myifft1)

print("Myfft - fft",np.linalg.norm(Myfft - _fft)) 
print("Myfft - Myifft",np.linalg.norm(x - Myifft1)) 
#
#print(np.linalg.norm(x - myifft(n, Myfft)))


# In[12]:


import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft 
n = 16
mytime=[]
myiffttime=[]
ffttime=[]
fftnorm=[]
ifftnorm=[]
for i in range(n):
    x =[]

    for j in range(2**i): 
        x.append(math.ceil(random.random()*10)) 
    start_time = time.clock() 
    Myfft = myfft(i,x) 
    mytime.append(time.clock()-start_time)

    start_time = time.clock() 
    _fft = fft(x) 
    ffttime.append(time.clock()-start_time)

    fftnorm.append(np.linalg.norm(Myfft - _fft)) 
    
    ifftnorm.append(np.linalg.norm(x - Difft(Myfft,i)))

    
    
print("fft norm",fftnorm)
print("ifftnorm",ifftnorm)

print("My fft time",mytime)
print("fft time",ffttime)
xx = np.arange(n)
xxx = 2**xx
fig = plt.figure()
plt.plot(xxx, mytime)

plt.plot(xxx, ffttime)


# In[54]:


B_n(2**2)


# In[62]:


16//4


# In[6]:


import numpy as np
n = 2**2
b = B_n(n)
m = len(b)
z = np.zeros(shape = (m,m), dtype = complex)
t = np.bmat([[b,z],[z,b]])
print(t)

