#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[63]:


#(d)
x_0 = np.array([0, 0])
f, g, H = my_objective(x, A, b, c)

newton_step = -np.dot(np.linalg.inv(H), g)
print(newton_step)


# In[7]:


#(d)~(j)
def newton_method(x, A, b, c):
    t = 1
    alpha = 0.1
    beta = 0.7
    iter_count = 0
    
    while(True):
        f, g, H = my_objective(x, A, b, c)
        newton_step = -np.dot(np.linalg.inv(H), g)
        newton_decrement = np.sqrt(-np.dot(g, newton_step))
        
        #stopping criteria
        if((np.square(newton_decrement)/2) <= 1e-10):
            print("stop at iteration =", iter_count+1)
            print("Iteration =", iter_count+1, ",f =", f)
            print("Iteration =", iter_count+1, ",decrement =", newton_decrement)
            return x
        
        print("Iteration =", iter_count+1, ",f =", f)
        print("Iteration =", iter_count+1, ",decrement =", newton_decrement)
        
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
    
def main():
    A = np.array([[1,3], [2,-3], [-1,0]])
    c = np.array([1, 1])
    b = np.array([0.2, 0.3, 0.4])
    x_0 = np.array([0, 0])
    
    x_sol = newton_method(x_0, A, b, c)
    f_sol = np.dot(c, x_sol)-math.log(-np.dot(A[0], x_sol)+b[0])-math.log(-np.dot(A[1], x_sol)+b[1])-math.log(-np.dot(A[2], x_sol)+b[2])
    print(x_sol, f_sol)

if __name__ == '__main__':
    main()


# In[3]:


import cvxpy as cp
import cvxpy

# Create two scalar optimization variables.
x = cp.Variable(2)

A = np.array([[1,3], [2,-3], [-1,0]])
c = np.array([1, 1]).reshape(2, 1)
b = np.array([0.2, 0.3, 0.4])

# Form objective.
obj = cp.Minimize(c.transpose()@x - cp.sum(cp.log((-A@x)+b)))

#obj = cp.Minimize(c.transpose()@x-cp.log((A[0].transpose()@x)+b[0])-cp.log((A[1].transpose()@x)+b[1])
                  #-cp.log((A[2].transpose()@x)+b[2]))

# Form and solve problem.
prob = cp.Problem(obj)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)


# In[ ]:




