#!/usr/bin/env python
# coding: utf-8

# In[100]:


import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x*(2-x)
def p(x):
    return x+1

def q(x):
    return -3

sigma1 = -1
mu1 = 0

sigma2 = 3
mu2 = -1

def _sigma1(x):
    return (sigma1 - q(x)*h/2 - sigma1*p(x)*h/2)

def _mu1(x):
    return (mu1 + f(x)*h/2 - mu1*p(x)*h/2)

def _sigma2(x):
    return (sigma2 + q(x)*h/2 + sigma2*h*p(x)/2)

def _mu2(x):
    return (mu2 - f(x)*h/2 + mu2*p(x)*h/2)



N = 10

h = 1/N

x = np.array([i*h for i in range (N+1)])

A = []
B = []
C = []
F = []

C.append(-(1/h + _sigma1(x[0])))
B.append(1/h)
F.append(_mu1(x[0]))


for i in range(1, N):
    A.append(1/h**2 - p(x[i-1])/(2*h))
    B.append( 1/h**2 + p(x[i])/(2*h))
    C.append(-((2/h**2) + 3))
    F.append(f(x[i]))

C.append(-(1/h + _sigma2(x[N])))
A.append(1/h)
F.append(_mu2(x[N]))

alpha = [0]
beta = [0]
def TrehDiag(a,b,c,f):

    n = len(f)
    x = [0]*n
    for i in range(n-1):
        alpha.append(-b[i]/(a[i-1]*alpha[i] + c[i]))
        beta.append((f[i] - a[i-1]*beta[i])/(a[i-1]*alpha[i] + c[i]))
    x[n-1] = (f[n-1] - a[n-2]*beta[n-1])/(c[n-1] + a[n-2]*alpha[n-1])
    for i in reversed(range(n-1)):
        x[i] = alpha[i+1]*x[i+1] + beta[i+1]
    return x

y = TrehDiag(A,B,C,F)
for i in range(N+1):
    print(i/10,y[i])
#print(y[int(N/2)])
#resid = np.zeros((N + 1))


# In[101]:


print("alpha",alpha)
print("beta",beta)


# In[81]:


-0.3202953850573808+0.3185355076740378


# In[ ]:




