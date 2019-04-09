#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
wiki RSA link
https://en.wikipedia.org/w/index.php?title=RSA_(cryptosystem)&oldid=891107390
Euclidean algorithm
https://ru.wikibooks.org/w/index.php?title=%D0%A0%D0%B5%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D0%B8_%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC%D0%BE%D0%B2/%D0%A0%D0%B0%D1%81%D1%88%D0%B8%D1%80%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%95%D0%B2%D0%BA%D0%BB%D0%B8%D0%B4%D0%B0&oldid=145525
'''
def bezout(a, b):
    x, xx, y, yy = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a % b
        x, xx = xx, x - xx*q
        y, yy = yy, y - yy*q
    return (x, y, a)

def RSA(mc, key):
    return pow(mc, key[0], key[1])


# In[5]:


p = 882493304303057 
q = 565640080106113
n = p*q
e = 435510454193522616856570224823
X_1 = 33938304564942541056706890572
Y_2 = 167669363821217143128176537107
d = 0
openkey =(e,n)
fi = (p-1)*(q-1)
#print(fi)
d =bezout(e,fi)[0]+fi
closedkey =(d,n)
X_2 = RSA(X_1,openkey)
X_3 = RSA(X_2,closedkey)
#print(closedkey)
print(X_1-X_3)
print(RSA(Y_2,closedkey))


# In[ ]:





# In[ ]:





# In[ ]:




