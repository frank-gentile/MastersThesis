#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:26:49 2020

@author: FrankGentile
"""
import numpy as np
import time

then = time.time()
n = 20
my_quiet = 0.5
my_tell = 0.5
quiet = 0.5
tell = 0.5
my_bias = [my_quiet, my_tell]
others_bias = [quiet,tell]
cooperation_table = np.zeros((2**n,n))
for i in range(1,(2**n)+1):
    for j in range(1,n+1):
        if i % (2**(n+1-j))>2**(n-j) or i % (2**(n+1-j))==0:
            cooperation_table[i-1][j-1]=1

transpose_cooperation_table = np.transpose(cooperation_table)
defectors = np.sum(cooperation_table,axis=1)

#define pain functions
defector_pain = np.log(1.5*defectors+0.01)
cooperator_pain = np.log(1.5*defectors+2.25)
dG = cooperation_table
#create pain table
for i in range(1,(2**n)+1):
    for j in range(1,n+1):
        if cooperation_table[i-1][j-1]==1:
            dG[i-1][j-1] = defector_pain[i-1]
        if cooperation_table[i-1][j-1]==0:
            dG[i-1][j-1] = cooperator_pain[i-1]

#create coefficients 
v = np.zeros((2**n+2*n,2**n))
for i in range(1,(2**n)+2*n+1):
    for j in range(1,(2**n)+1):
        if i<=2*n and i % 2 == 1 and j % 2**(n+1-(i+1)/2) <= 2**(n-(i+1)/2) and j % 2**(n+1-(i+1)/2) > 0:
            v[i-1][j-1]=-1
        if i<=2*n and i % 2 == 0 and j % 2**(n+1-i/2) > 2**(n-(i)/2) and j % 2**(n+1-(i)/2) > 0:
            v[i-1][j-1]=-1
        if i<=2*n and i % 2 == 0 and j % 2**(n+1-i/2)==0:
            v[i-1][j-1]=-1
        if i>2*n and i - 2*n == j:
            v[i-1][j-1]=1

decider_v = np.zeros(((2**n)*(n+1),2**n))
for i in range(1,((2**n)*(n+1))+1):
    for j in range(1,(2**n)+1):
        if i % 2**n == j and i <= (2**n)*n:
            decider_v[i-1][j-1]=-1
        if i % 2**n == 0 and i <= (2**n)*n and j == 2**n:
            decider_v[i-1][j-1]=-1
        if i % 2**n == j and i > (2**n)*n:
            decider_v[i-1][j-1]=1
        if i % 2**n == 0 and i>(2**n)*n and j == 2**n:
            decider_v[i-1][j-1]=1


#create bias matrix
my_bias_matrix = np.zeros(((2**n,n)))
for i in range((2**n)+1):
    for j in range(n+1):
        if i % 2**(n+1-j) > 2**(n-j) or i % 2**(n+1-j)==0:
            my_bias_matrix[i-1][j-1] = my_tell
        else: my_bias_matrix[i-1][j-1] = my_quiet
others_bias_matrix = np.zeros(((2**n,n)))
for i in range((2**n)+1):
    for j in range(n+1):
        if i % 2**(n+1-j) > 2**(n-j) or i % 2**(n+1-j)==0:
            others_bias_matrix[i-1][j-1] = tell
        else: others_bias_matrix[i-1][j-1] = quiet
            
        
bias = my_bias + others_bias*(n-1)
total_bias = sum(bias)

others_bias_matrix_prod = np.prod(others_bias_matrix,axis=1)
ex = np.transpose(np.exp(-dG))/total_bias**(n-1)*others_bias_matrix_prod/np.transpose(others_bias_matrix)*np.transpose(my_bias_matrix)
extent = np.transpose(ex)
sum_ex = sum(sum(ex))

exx = np.exp(1)/sum_ex**(n-1)*np.prod(ex,axis=0)


change = np.dot(v,extent)
yDecider = exx/sum(exx)
cooperation_rate_a = sum(change[0])/(sum(change[0])+sum(change[1]))
now = time.time()
print("Cooperation rate ="+str(cooperation_rate_a))
print("Time: ", now-then, "seconds" )

