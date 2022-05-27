#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import math

#(c)
def my_objective(x, A, b, c):
    obj = np.dot(c, x)-math.log(-np.dot(A[0], x)+b[0])-math.log(-np.dot(A[1], x)+b[1])-math.log(-np.dot(A[2], x)+b[2])
    
    gradient = np.array([c[0]-A[0][0]/(np.dot(A[0], x)-b[0])-A[1][0]/(np.dot(A[1], x)-b[1])-A[2][0]/(np.dot(A[2], x)-b[2]),
                        c[1]-A[0][1]/(np.dot(A[0], x)-b[0])-A[1][1]/(np.dot(A[1], x)-b[1])-A[2][1]/(np.dot(A[2], x)-b[2])])
    
    hessian = np.array([[A[0][0]*A[0][0]/pow(np.dot(A[0],x)-b[0],2)+A[1][0]*A[1][0]/pow(np.dot(A[1],x)-b[1],2)+A[2][0]*A[2][0]/pow(np.dot(A[2],x)-b[2],2),
                         A[0][0]*A[0][1]/pow(np.dot(A[0],x)-b[0],2)+A[1][0]*A[1][1]/pow(np.dot(A[1],x)-b[1],2)+A[2][0]*A[2][1]/pow(np.dot(A[2],x)-b[2],2)],
                         [A[0][1]*A[0][0]/pow(np.dot(A[0],x)-b[0],2)+A[1][1]*A[1][0]/pow(np.dot(A[1],x)-b[1],2)+A[2][1]*A[2][0]/pow(np.dot(A[2],x)-b[2],2),
                         A[0][1]*A[0][1]/pow(np.dot(A[0],x)-b[0],2)+A[1][1]*A[1][1]/pow(np.dot(A[1],x)-b[1],2)+A[2][1]*A[2][1]/pow(np.dot(A[2],x)-b[2],2)]])
    
    return obj, gradient, hessian


# In[35]:


import numpy as np
import math
import matplotlib.pyplot as plt
import cvxpy as cp

#(d)~(j)
def newton_method(x, A, b, c):
    t = 1
    alpha = 0.1
    beta = 0.7
    iter_count = 0
    decrement_list = []
    
    while(True):
        f, g, H = my_objective(x, A, b, c)
        newton_step = -np.dot(np.linalg.inv(H), g)
        newton_decrement = np.sqrt(-np.dot(g, newton_step))
        
        #stopping criteria
        if((np.square(newton_decrement)/2) <= 1e-10):
            print("stop at iteration =", iter_count+1)
            print("Iteration =", iter_count+1, ",f =", f)
            print("Iteration =", iter_count+1, ",decrement =", newton_decrement)
            decrement_list.append(newton_decrement)
            print("----------------------------------------")
            print("decrement_list = ", decrement_list)
            index_list = []
            for i in range(0, iter_count+1):
                index_list.append(i)
    
            plt.figure(figsize = (5, 3), dpi = 100)
            plt.xlabel('k', fontsize = 12)
            plt.ylabel('(decrement^2)/2', fontsize = 12)
            plt.xticks(np.arange(0, iter_count+2, step=1), fontsize = 10)
            plt.yticks(fontsize = 10)

            plt.plot(index_list, np.square(decrement_list)/2, color = 'red', linewidth = 2)
            plt.show()
    
            return x
        
        print("Iteration =", iter_count+1, ",f =", f)
        print("Iteration =", iter_count+1, ",decrement =", newton_decrement)
        decrement_list.append(newton_decrement)
        
        #backtracking line search to choose t
        x_next = x + t * newton_step
        
        #check dom f
        while((-np.dot(A[0], x_next)+b[0]<0) or (-np.dot(A[1], x_next)+b[1]<0) or (-np.dot(A[2], x_next)+b[2]<0)):
            t = beta * t
            x_next = x + t * newton_step
            print("Out of dom f!")
            
        print("Iteration =", iter_count+1, ",t =", t)
        print("----------------------------------------")
        
        f_next = np.dot(c, x_next)-math.log(-np.dot(A[0], x_next)+b[0])-math.log(-np.dot(A[1], x_next)+b[1])-math.log(-np.dot(A[2], x_next)+b[2])
        while(f_next > (f - alpha * t * np.square(newton_decrement))):
            t = beta * t
            x_next = x + t * newton_step
            f_next = np.dot(c, x_next)-math.log(-np.dot(A[0], x_next)+b[0])-math.log(-np.dot(A[1], x_next)+b[1])-math.log(-np.dot(A[2], x_next)+b[2])
        
        x = x + t * newton_step
        iter_count =  iter_count + 1

def cvx_tool(A, c, b):
    # Create two scalar optimization variables.
    x = cp.Variable(2)
    c.reshape(2, 1)

    # Form objective.
    obj = cp.Minimize(c.transpose()@x - cp.sum(cp.log((-A@x)+b)))

    # Form and solve problem.
    prob = cp.Problem(obj)
    prob.solve()  # Returns the optimal value.
    print("status:", prob.status)
    print("CVX optimal value", prob.value)
    print("CVX optimal var", x.value)        
    
def main():
    print("Setting 1:")
    A = np.array([[1,3], [1,-3], [-1,0]])
    c = np.array([1, 2])
    b = np.array([0.1, 0.1, 0.1])
    x_0 = np.array([0, 0])
    
    x_sol = newton_method(x_0, A, b, c)
    f_sol = np.dot(c, x_sol)-math.log(-np.dot(A[0], x_sol)+b[0])-math.log(-np.dot(A[1], x_sol)+b[1])-math.log(-np.dot(A[2], x_sol)+b[2])
    print("Newton's method: optimal value", f_sol)
    print("Newton's method: optimal var", x_sol)
    print("---------Compare to CVX---------")
    
    cvx_tool(A, c, b)
    
    print("\nSetting 2:")
    A = np.array([[1,3], [2,-3], [-1,0]])
    c = np.array([1, 1])
    b = np.array([0.2, 0.3, 0.4])
    x_0 = np.array([0, 0])
    
    x_sol = newton_method(x_0, A, b, c)
    f_sol = np.dot(c, x_sol)-math.log(-np.dot(A[0], x_sol)+b[0])-math.log(-np.dot(A[1], x_sol)+b[1])-math.log(-np.dot(A[2], x_sol)+b[2])
    print("Newton's method: optimal value", f_sol)
    print("Newton's method: optimal var", x_sol)
    print("---------Compare to CVX---------")
    
    cvx_tool(A, c, b)
    
if __name__ == '__main__':
    main()


# In[9]:


import cvxpy as cp
import cvxpy

# Create two scalar optimization variables.
x = cp.Variable(2)

A = np.array([[1,3], [1,-3], [-1,0]])
c = np.array([1, 2]).reshape(2, 1)
b = np.array([0.1, 0.1, 0.1])

# Form objective.
obj = cp.Minimize(c.transpose()@x - cp.sum(cp.log((-A@x)+b)))

# Form and solve problem.
prob = cp.Problem(obj)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)


# In[ ]:





# In[14]:


a = [2, 5]
np.square(a)/2


# In[ ]:




