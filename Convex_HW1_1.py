#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cvxpy as cp

# Create two scalar optimization variables.
x = cp.Variable()
y = cp.Variable()

# Create constraints.
constraints = [pow(x,-1) + pow(y,-1) -1 <= 0, x >= 0, y >= 0]

# Form objective.
obj = cp.Minimize(2*x+y)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)


# In[31]:


1/1.7071422256133406 + 1/2.414142541321234


# In[32]:


2*1.7071422256133406+2.414142541321234


# In[ ]:




