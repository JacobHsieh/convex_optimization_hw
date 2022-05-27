#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cvxpy as cp

#Bisection method
l =  0.0
u =  0.3
toleran = 0.00000001
c = 4

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
print(x.value[0]+c*x.value[1])
print((x.value[0]**2+x.value[1]**2)**0.5-1/(2**0.5))


# In[24]:


import numpy as np
import matplotlib.pyplot as plt

ans = np.array([0.250000207722457, 0.2500000171259619, 0.1250000142615753, 0.08333335260723379, 0.06250001277387675])
c = np.array ([0.5, 1, 2, 3, 4])

plt.figure(figsize = (5, 3), dpi = 100)
plt.xlabel('parameter c', fontsize = 12)
plt.ylabel('optimal value', fontsize = 12)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)

plt.plot(c, ans, color = 'red', linewidth = 2)
plt.show()


# In[ ]:




