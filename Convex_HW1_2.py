#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cvxpy as cp
import numpy as np
from cvxpy import norm

# Create two scalar optimization variables.
x = cp.Variable(5)
y = np.array([1, 2, 3, 4, 5]).reshape(5, 1)

# Create constraints.
constraints = [x >= 0, pow(x[0],-1)+pow(x[1],-1)+pow(x[2],-1)+pow(x[3],-1)+pow(x[4],-1) <= 1]

# Form objective.
obj = cp.Minimize(norm(x, 3) + y.transpose() @ x)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)


# In[3]:


import cvxpy as cp
import numpy as np
from cvxpy import norm
import matplotlib.pyplot as plt

# Create two scalar optimization variables.
x = cp.Variable(5)
y = np.array([1, 2, 3, 4, 5]).reshape(5, 1)

# Create two constraints.
constraints = [x >= 0, pow(x[0],-1)+pow(x[1],-1)+pow(x[2],-1)+pow(x[3],-1)+pow(x[4],-1) <= 1]

ans = np.zeros((100))
p = np.arange(1, 101)

# Form objective.
for i in range(1, 101):
    obj = cp.Minimize(norm(x, i) + y.transpose() @ x)
    prob = cp.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    ans[i-1] = prob.value

plt.figure(figsize = (5, 3), dpi = 100)
plt.xlabel('parameter p', fontsize = 12)
plt.ylabel('optimal value', fontsize = 12)
plt.xticks(np.arange(0, 101, step=10), fontsize = 10)
plt.yticks(fontsize = 10)
#plt.annotate('min_acc', xy=(min_a_u1, val_u1(min_a_u1)), xytext=(min_a_u1-0.5, val_u1(min_a_u1)), arrowprops=dict(facecolor='k', headwidth=5, width=2))

plt.plot(p, ans, color = 'red', linewidth = 2)
plt.show()


# In[4]:


ans[2]


# In[ ]:




