#!/usr/bin/env python
# coding: utf-8

# In[25]:


import hashlib
import random
q = 210697455032337684943121194039863591186004713463570796268765689709223108292419

def bezout(a, b):
    x, xx, y, yy = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a % b
        x, xx = xx, x - xx*q
        y, yy = yy, y - yy*q
    return (x, y, a)

def Gen(q):
    while 1:
        R = 2*random.randint(1,2*(q+1)) # R<4(q+1)
        p = q*R+1
        if pow(2, q*R, p)==1 and pow(2, R, p)!=1:
            break
    x = random.randint(1,p)
    while 1:
        g = pow(x, R, p)
        if g!=1:
            break
    d = random.randint(1,g)
    e = pow(g, d, p)
    return((p,q,g),e,d)

def Sign(pqg, d, M):
    p = pqg[0]
    q = pqg[1]
    g = pqg[2]
    m = int(hashlib.sha256(M).hexdigest(),16)   
    k = random.randint(1,q-1)
    r = pow(g,k,p)
    k_1=bezout(k,q)[0]
    s= k_1*(m-d*r)%q
    return(r,s)

def Verify(pqg, e, M, rs):
    p = pqg[0]
    q = pqg[1]
    g = pqg[2]
    r = rs[0]
    s = rs[1]
    if (r<0 or r>p) or (s<0 or s>q):
        return False
    m = int(hashlib.sha256(M).hexdigest(),16)   
    if (pow(e,r,p)*pow(r,s,p)-pow(g,m,p))%p == 0:
        return True
    return False
    
A = Gen(q)
#print(A)
rs = Sign(A[0],A[2],b"Hello World")
print(Verify(A[0],A[1],b"Hello World",rs))
print(Verify(A[0],A[1],b"World",rs))


# In[2]:


def fast_pow(x, y, z):
    if y == 0:
        return 1
    if y == -1:
        return 1. / x
    p = fast_pow(x, y / 2, z)%z
    p *= p
    p = p%z
    if y % 2:
        p *= x%z
    return p%z
print(pow(15,7, 4))
print(fast_pow(15,7,4))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




