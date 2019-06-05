#!/usr/bin/env python
# coding: utf-8

# In[224]:


import numpy as np
import matplotlib.pyplot as plt

def TrehDiag(a,b,c,f, j):
    alpha = [0]
    beta = [0]
    n = len(f)
    x = [0]*n
    print(f[0])

    f[0] =  0.6#  (h/(2*tu**2))*(-2*ys[j, 0]   +ys[j-1,0]) - 0.6 - func(0, tu*j)*h/2-(1-2*sigma)*(y_x_rlt(1,j))-sigma*(y_x_rlt(1, j-1))
    for i in range(1, n-1):
        f[i] = -(2*ys[j, i]/tu**2 - ys[j-1, i]/tu**2 + (1-2*sigma)*y_xx(i, j) + sigma*y_xx(i, j-1) + func(i*h, tu*j))
    #f[n-1] = (ys[j-1, n-1] - 2*ys[j, n-1])*h/(2*tu**2) - np.cos(2) - func(1, t[j])*h/2 - (1-2*sigma)*y_x_left(n-1, j) - sigma*y_x_left(n-1, j-1)
    f[n-1] = -((h/(2*tu**2))*(-2*ys[j, n-1] + ys[j-1, n-1]) - 0.6*np.cos(2) - func(1, tu*j)*h/2) - (1-2*sigma)*y_x_left(n-1, j) - sigma*y_x_left(n-1,j-1)
    
    for i in range(n-1):
        alpha.append(-b[i]/(a[i-1]*alpha[i] + c[i]))
        beta.append((f[i] - a[i-1]*beta[i])/(a[i-1]*alpha[i] + c[i]))
    x[n-1] = (f[n-1] - a[n-2]*beta[n-1])/(c[n-1] + a[n-2]*alpha[n-1])
    for i in reversed(range(n-1)):
        x[i] = alpha[i+1]*x[i+1] + beta[i+1]
    return x

def func(x, t):
    return -1.8*np.cos(3*t) + 1.2*np.sin(x)

def y_x0(x):
    return 0.3*(np.sin(2*x) + 2)

def y_0t(t):
    return (0.2*np.cos(3*t) + 0.3)

def y_tt(i, j):
    return (ys[j+1, i] - 2*ys[j, i] + ys[j-1, i])/tu**2

def y_x_left(i, j):
    return (ys[j, i] - ys[j, i-1])/h

def y_x_rlt(i, j):
    return (ys[j,1]-ys[j,0])/h

def y_xx(i, j):
    return (ys[j, i+1] - 2*ys[j, i] + ys[j, i-1])/h**2

T = 0.5
Nx = 100

h = 1/Nx
tu = 0.05

Nt = int(T/tu)

sigma = 1/2

x = np.array([i*h for i in range (Nx+1)])
t = np.array([i*tu*T for i in range (Nt+1)])



ys = np.zeros((Nt + 1, Nx + 1))


ys[0] = y_x0(x)

ys[1] = y_x0(x)

A = np.zeros((Nx))
B = np.zeros((Nx))
C = np.zeros((Nx + 1))
F = np.zeros((Nx + 1))

C[0] = 1/h#(-sigma/h-h/(2*tu**2))
B[0] = -1/h#(sigma/h)

C[Nx] = sigma/h+h/(2*tu**2)
A[Nx - 1] = -sigma/h

for i in range(1, Nx):
    A[i-1] = sigma/h**2
    B[i] = sigma/h**2
    C[i] = -(1/tu**2 + 2*sigma/h**2)

for j in range(1, Nt):
    ys[j+1] = TrehDiag(A,B,C,F,j)

plt.plot(x, ys[1])
plt.plot(x, ys[2])
plt.plot(x, ys[3])
plt.show()


# In[225]:


for i in range(11):
    print("time", i*tu,ys[i])


# In[208]:


print(ys[3])


# In[ ]:





# In[ ]:




