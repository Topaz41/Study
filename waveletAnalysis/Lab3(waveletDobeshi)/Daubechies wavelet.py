#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import math
N = 5
def Cf_k(n, k):
    return math.factorial(n)/(math.factorial(k)*(math.factorial(n-k)))
    


# In[6]:


Q_ymas =[]
for k in range(N):
    Q_ymas.append(Cf_k(N-1+k,k))
print(Q_ymas)    


# In[ ]:


#-y^5 + 5 y^4 - 10 y^3 + 10 y^2 - 5 y + 1
#1.0 + 5.0y+ 15.0y^2+ 35.0y^3+ 70.0y^4
#-70 y^9 + 315 y^8 - 540 y^7 + 420 y^6 - 126 y^5 + 1


# In[173]:


#(1.0 + 5.0(1-cosx)/2)+ 15.0((1-cosx)/2)^2+ 35.0((1-cosx)/2)^3+ 70.0((1-cosx)/2)^4)
#-(1/2+cosx)^5 + 5 1/2+cosx^4 - 10 1/2+cosx^3 + 10 1/2+cosx^2 - 5 1/2+cosx + 1
def column(matrix, i):
    return (row[i] for row in matrix)
def Haar(x):
    #print(x)
    if (x<0):
        return 0;
    if (x>=1):
        return 0
    return 1


# In[293]:


h = [(1+pow(3,1/2))/(4*pow(2,1/2)),(3+pow(3,1/2))/(4*pow(2,1/2)),(3-pow(3,1/2))/(4*pow(2,1/2)),(1-pow(3,1/2))/(4*pow(2,1/2))]
#h = [0.3326705529500825,0.8068915093110924,0.4598775021184914,-0.1350110110200102546,-0.0854412738820267,0.0352262918867095]
#h = [0.0380779473638778, 0.2438346746125858, 0.6048231236900955, 0.6572880780512736, 0.1331973858249883, -0.2932737832791663, -0.0968407832229492, 0.1485407493381256, 0.0307256814793385, -0.0676328290613279, 0.0002509471148340, 0.0223616621236798, -0.0047232047577518, -0.0042815036824635, 0.0018476468830563, 0.0002303857635939, 0.0002519631889427, 0.0000393473203163]
n = int(len(h))
pointCount = 61
iter = 24
x = np.zeros(pointCount)

xxx = np.zeros(pointCount)
for i in range(len(xxx)):
    xxx[i] =  1/((pointCount-1)/6)*(i-(pointCount-1)/2)
x= xxx

#for i in range(pointCount):
#    x[i] = 1/((pointCount-1)/3)*i
y = np.zeros([pointCount, iter])
def point(x, i):
    for j in range(len(x)):
        if(abs(x[j]-i)<0.0005):
            return j
    print("kek")
    return(0)
    

for i in range(pointCount):
    temp =0
    for j in range(n):
        temp = temp + h[j]*Haar(2*x[i]-j)
    y[i][0]= temp*pow(2,1/2)

for k in range(1,iter):
    for i in range(pointCount):
        temp = 0
        for j in range(n):
            arg = 2*x[i]-j
            if arg >=-3 and arg<=3:
                #print(arg)
                temp = temp + h[j]*y[point(x,arg)][k-1]
        y[i][k]= temp*pow(2,1/2)
fi = y[:,iter-1]

plt.plot(x,fi)


# In[294]:


psi =  np.zeros(pointCount)
for i in range(len(x)):
    temp = 0
    for j in range(-10 , 10):
        arg = 2*x[i]-j        
        
        if -j+1>=0 and -j+1 < n:
            q_n = (-1)**j*h[-j+1]
            if arg >= -3 and arg<=3:                
                temp = temp + q_n*fi[point(x, arg)]
            
    psi[i]= temp
    
    
plt.plot(x,psi)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




