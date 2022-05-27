#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cvxpy as cp
from cvxpy import norm

x = cp.Variable(2, pos = True)
l = -1
u = 1
toleran = 0.0001


t = 1/2*(l+u)
x = cp.Variable(2, pos=True)
    
constraints = [x[0]+2*x[1] <= 1, norm(x) <= 1/(2**0.5)]
    
obj = cp.Maximize((x[0] * x[1]))
    
prob = cp.Problem(obj, constraints)
prob.solve(qcp = True)

        
print("optimal value", prob.value)
print("optimal var", x.value)


# In[ ]:




