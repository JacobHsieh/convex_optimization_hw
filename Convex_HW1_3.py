#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cvxpy as cp
from cvxpy import norm

# Create two scalar optimization variables.
x = cp.Variable(pos=True)
y = cp.Variable(pos=True)
    
# Create constraints.
constraints = [x+2*y <= 1, pow(x, 2)+pow(y, 2) <= 1/2]
    
# Form objective.
#obj = cp.Minimize(-cp.multiply(x, y))
obj = cp.Maximize(cp.multiply(x, y))
    
# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve(qcp = True, verbose = True)  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)
    


# In[7]:


0.49999755881244756*0.2500012206164112


# In[ ]:




