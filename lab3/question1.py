from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sp
import numpy as np
import seaborn as sb

S = np.array([[1,2,3,4], [0,1,0,1], [1,4,3,6], [1,11,6,15]])

vec_in_s = np.array([1,0,0,0])
basis = sp.linalg.orth(np.transpose(S)) # Orthonormal basis
basis = np.transpose(basis)

projected_vec = np.array([0,0,0,0])
for basis_vec in basis:
    projected_vec = projected_vec + basis_vec*(np.dot(vec_in_s,basis_vec))

if np.array_equal(np.isclose(projected_vec, vec_in_s), [1,1,1,1]):
    print "Vector " + str(vec_in_s) + " is in S."
else:
    print "Vector " + str(vec_in_s) + " is not in S."
    print projected_vec

print "Dimension of S is " + str(basis.shape[0]) + "."

print "Orthonormal basis for S:"
for basis_vec in basis:
    print str(basis_vec)
