#!/usr/bin/env python
# coding: utf-8

# In[198]:


import numpy as np
import random
import time
from scipy.fftpack import fft, ifft
from scipy import linalg
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')
#import pylab
import matplotlib.patches
import matplotlib.lines
import matplotlib.path
from sympy import diff, symbols, cos, sin

a=1
#du_dt1 = -a*sin(a*t - 2*pi*x)
#u_dx1 = 2*pi*sin(a*t - 2*pi*x)
#du_dxLdu_dx1 = -4*pi**2*(-x + 1.01)*cos(a*t - 2*pi*x) - 2*pi*sin(a*t - 2*pi*x)
#ff1 = du_dt1 - du_dxLdu_dx1
l = 1
T=1

tau = 0.0025
h=1/10
sigma = 1
w,q = int(l/h), int(T/tau);

Matrix = [[0 for x in range(q+1)] for y in range(w+1)] 

for i in range(w):
    Matrix[i][0] = math.sin(i*h)
for j in range(q+1):
    #Matrix[0][j] =0
    Matrix[w][j] = math.sin((j*tau+1))

#print(Matrix)
#lam = 1


# In[199]:


a=1
def Lamb(x):
    return 1;
def ff(x, t):
    return math.pow(2,1/2)*math.sin(math.pi/4+x+t)
def TrehDiag(a,b,c,f):
    alpha = [0]
    beta = [0]
    n = len(f)
    x = [0]*n
    for i in range(n-1):
        alpha.append(-b[i]/(a[i-1]*alpha[i] + c[i]))
        beta.append((f[i] - a[i-1]*beta[i])/(a[i-1]*alpha[i] + c[i]))
    x[n-1] = (f[n-1] - a[n-2]*beta[n-1])/(c[n-1] + a[n-2]*alpha[n-1])
    for i in reversed(range(n-1)):
        x[i] = alpha[i+1]*x[i+1] + beta[i+1]
    return x


# In[200]:


a=1
for j in range (q):
    aa=[]
    b=[]
    b.append(sigma/h)
    c=[]
    c.append(-sigma/h-2*sigma-h/(2*tau))
    f=[]
    f.append(-(math.cos(tau*j)-2*math.sin(tau*j)+h/2*pow(2,1/2)*math.sin(math.pi/4+tau*j))-Matrix[1][j]*((1-sigma)/h)-Matrix[0][j]*(-(1-sigma)/h-2*(1-sigma)+h/(2*tau)))
    #print(j,"j")
    for i in range (1,w):
        #print(ff(i*h,j*h))
        #print(i,"i")
        #f.append(-(1/tau*Matrix[i][j]+(((1-sigma)/h**2)*(lamb(i*h) - lamb((i+1)*h))*((Matrix[i][j])-(Matrix[i-1][j])) + lamb((i+1)*h)*((Matrix[i+1][j])-2*Matrix[i][j]+Matrix[i-1][j]))+sigma*ff(i*h,(j+1)*h)+(1-sigma)*ff(i*h,j*h)))
        #b.append(-sigma*(lamb((i+1)*h))/h**2)
        #c.append((1/tau-(sigma/h**2)*(lamb((i+1)*h)-lamb(i*h))-2*sigma*lamb((i+1)*h)/h**2))
        #aa.append(-(sigma/h**2)*(lamb((i+1)*h)-lamb(i*h)) - (sigma/h**2)*(lamb((i+1)*h)))  
        f.append(-((1-sigma)*Lamb((i-1/2)*h)/h**2*Matrix[i-1][j]+((1-sigma)*(Lamb((i+1/2)*h)+Lamb((i-1/2)*h))/(-h**2)+1/tau)*Matrix[i][j]+(1-sigma)*Lamb((i+1/2)*h)/h**2*Matrix[i+1][j]+sigma*ff(i*h,(j+1)*tau)+(1-sigma)*ff(i*h,j*tau)))
        aa.append(sigma*Lamb((i-1/2)*h)/h**2)
        c.append(-(sigma*(Lamb((i+1/2)*h)+Lamb((i-1/2)*h))/h**2+1/tau))
        b.append(sigma*Lamb((i+1/2)*h)/h**2)
    aa.append(0)
    c.append(1)
    f.append(Matrix[-1][j+1])
    #print(f)
    #aa = np.array(aa)
    #mat = np.diag(aa, -1) + np.diag(c, 0) + np.diag(b, 1)
    #print(mat,f)
    ga = TrehDiag(aa,b,c,f)#np.linalg.solve(mat, f)
    #print(mat,f)
    for i in range (w+1):
        Matrix[i][j+1] = ga[i]
    #print(ga)
        


# In[201]:


dfd = np.array(Matrix)
k=40
for i in range(11):
    print("time",0.1*i)
    print("res",dfd[:,i*k])


# In[142]:


dfd = np.array(Matrix)
print(dfd[:,1])


# In[143]:


dfd = np.array(Matrix)
xxx = [i*h for i in range(w+1)]
y1 = (dfd[:,0])
y2 = (dfd[:,4])
y3 = (dfd[:,99])
#print(y1,y2,y3)
plt.plot(xxx, y1)
plt.plot(xxx, y2)
plt.plot(xxx, y3)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




