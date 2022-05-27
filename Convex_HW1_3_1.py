#!/usr/bin/env python
# coding: utf-8

# In[105]:


import cvxpy as cp
import numpy as np

#Bisection method
for c in [0.5, 1, 2, 3, 4]:
    l =  0.0
    u =  0.3
    toleran = 0.00000001
    ans = np.zeros(5)
    i = 0
    
    while(u-l >= toleran):
        t = 1/2*(l + u)
        x = cp.Variable(2, pos = True)
        
        constraints = [x[0]+c*x[1] <= 1, cp.norm(x, 2) <= 1/(2**0.5), x[0] * x[1] >= t]
        
        obj = cp.Maximize(x[0] * x[1])
        
        prob = cp.Problem(obj, constraints)
        prob.solve(qcp=True)
        
        if(prob.status == "infeasible"):
            u = t
        else:
            l = t
        
    print("c=", c)
    print("optimal value=", prob.value)
    print("optimal var", x.value)
    print(x.value[0]+c*x.value[1]-1)
    print((x.value[0]**2+x.value[1]**2)**0.5-1/(2**0.5))
    ans[i] = prob.value
    i += 1
    


# In[101]:


i


# In[90]:


0.50004113+2*0.24997947


# In[91]:


(0.50004113**2+0.24997947**2)**0.5


# In[93]:


pow(pow(0.50004113,2)+pow(0.24997947,2),0.5)


# In[92]:


1/(2**0.5)


# In[ ]:


import numpy as np

ans = np.array([0.1250000142615753, ])
c = np.array ([0.5, 1, 2, 3, 4])


# In[97]:


for c in [0.5, 1, 2, 3, 4]:
    print(c)


# In[103]:


ans = np.zeros(5)
ans


# In[ ]:




