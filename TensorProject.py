import numpy as np
import pandas as pd
import scipy as sc
import math
from sklearn.preprocessing import Imputer
from fancyimpute import IterativeSVD, BiScaler, KNN, NuclearNormMinimization, SoftImpute

import tensorflow as tf

def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

def TensorProj(M):

    le = len(M.T)
    N = len(M)
    print('we entereted to TensorProj func\n')
  #  T = [[[0 for i in range(N)] for j in range(N)] for k in range(N)]
 #   T = tf.zeros([len(e1),len(e1),len(e1)], tf.float32)
 #   print('empty Twas created\n')
    with open('T.text','a') as fout:
        fout.write('[')
    for i in range(0,N):
        with open('T.text','a') as fout:
            fout.write('[')
        for j in range(0,N):
            with open('T.text','a') as fout:
                fout.write('[')
            for k in range(0,N):
                x=0
                for l in range(0,le):
                    x += M[i,l] * M[j,l]* M[k,l]
                with open('T.text','a') as fout:
                    fout.write(str(x) + ',')
 #                   T[i][j][k] += M[i,l] * M[j,l]* M[k,l]
            with open('T.text','a') as fout:
                fout.write(']')
        with open('T.text','a') as fout:
            fout.write(']')
        print(i)
    with open('T.text','a') as fout:
        fout.write(']')
    fout.close()
    T = np.loadtxt('T.txt')
    tensor = my_func(T)
    print('Tensor is created\n')
    return tensor

def decom(T,g):

    X = tf.Variable(tf.random_normal([N,N]))
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    print(sess.run(X))
 #   V = tf.Variable(tf.random_normal([N]))
    #Power method
#    for i in range(5):
    X = tf.einsum('abi,cdj,ijk,k,bd->ac',T,T,T,g,X)
    
 #   X = tf.einsum('ijk,k,j',T,g,V)
 #       norm = tf.nn.l2_normalize(X, 0, epsilon = 1e-12, name = None)
  #      norm = tf.norm(X)
 #       print(sess.run(X))
 #       X = X / norm
 #       maxVal = tf.reduce_max(X)
 #       X = X/maxVal
 #       X = tf.abs(X)

    #sigma = tf.reduce_sum(X,0)
    #print(sess.run(sigma))
    #X = tf.multiply(X,sigma)
    
    E = tf.self_adjoint_eig(X, name = None)
 #   print('\n eigen decomposition: \n')
#    print(sess.run(E[0]))
#    print("\n")
#    print(sess.run(E[1]))
    
    #Finding top eigenvector
    with sess.as_default():
        e0 = E[0].eval()
#        maximum = e0[0]
#        for i in range(len(e0)):
#            if e0[i] > maximum:
#                maximum = e0[i]
#                MaxIndex = i

 #   eigenMax = maximum
    eigenMax = e0[N-1]
    MaxIndex = N-1
    print("\n top eigenvalue \n")
    print(eigenMax)
    print("\n top eigenvector\n")
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    print(sess.run(E[1][:,MaxIndex]))

    print("\n max index \n")
    print(MaxIndex)
        
    v1 = E[1][:,MaxIndex]
    T1 = my_func(v1)
    #norm = tf.nn.l2_normalize(T1, 0, epsilon = 1e-12, name = None)
    #T1 = norm
    #T1 = tf.abs(T1)   
    
    decomp = tf.einsum ('i,j,k->ijk', T1, T1, T1)
 #   decomp = tf.einsum('ij,jk->ik', T1, T1)
 #   T = tf.einsum(alpha,decomp)
 #   T = tf.multiply(T1,X)
 #   T = tf.einsum('i,jk',T1,decomp)
    T = MaxIndex * decomp
    print("\n")
    print("Tesnor decomp result\n")
 #   print(sess.run(T))

    return T

def main():

    #Creating matrix e1
            
    sig_matrix = np.loadtxt('Clear_Cell_Cycle.txt')
    print("Dimension of raw e1: ")
    print(sig_matrix.shape)
    print('\n')

#---------------------------------------------------
    
    #Filling missing data in e1
    
    sig_matrix[sig_matrix == 0] = np.NaN
    X_incomplete = sig_matrix
    #imputer = Imputer()
    #transformed_sig_matrix = imputer.fit_transform(sig_matrix)
    #Count the number of NaN values in each column
    #print(np.isnan(transformed_sig_matrix).sum())
    #sig_matrix = transformed_sig_matrix

    # Use SVD
    X_filled = IterativeSVD().complete(X_incomplete)
    # Use 3 nearest rows which have a feature to fill in each row's missing features
    X_filled_knn = KNN(k=5).complete(X_incomplete)
 #   svd_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
 #   print("IterativeSVD MSE: %f" % svd_mse)
    
    # matrix completion using convex optimization to find low-rank solution
    # that still matches observed values. Slow!
    #X_filled_nnm = NuclearNormMinimization().complete(X_incomplete)

    # print mean squared error for the three imputation methods above
    #nnm_mse = ((X_filled_nnm[missing_mask] - X[missing_mask]) ** 2).mean()
    #print("Nuclear norm minimization MSE: %f" % nnm_mse)

 #   knn_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
 #   print("knnImpute MSE: %f" % knn_mse)

    sig_matrix = X_filled_knn
    e1 = sig_matrix

 #   print("\n Tesnor 1 before decomposition\n")
    #with open("Tensor_Before_Decomposition", 'w') as fout:
        #fout.writelines(sess.run(T))
   # print(sess.run(T))
    Tfirst = T = TensorProj(e1)
    n = tf.norm(T)
    print("\n")
    print(sess.run(n))
    T_new = tf.zeros([len(e1),len(e1),len(e1)], tf.float32)
    
    #scaling T
 #   norm = tf.nn.l2_normalize(T, 0, epsilon = 1e-12, name = None)
 #   norm = tf.norm(T)
 #   T = norm
#    print(sess.run(T))

 #   fout = open("norm.txt", 'a')
    for i in range(200):
        
        decomp = decom(T,g)
        T_new += decomp
        T = decomp
        decomp = Tfirst - T_new
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        n = tf.norm(decomp)
        print("\n")
        print(sess.run(n))
        with open('norm.text','a') as fout:
            fout.write(str(sess.run(n)) + '\n')
        fout.close()
        
    #norm = tf.nn.l2_normalize(T_new, 0, epsilon = 1e-12, name = None)
    #print("\n sum of decomposition:\n")
    #init_op = tf.global_variables_initializer()
    #sess = tf.Session()
    #sess.run(init_op)
    #print(sess.run(T_new))
    print("\n")

    Tfirst = T = TensorProj(e2)

    print("Tesnor 2 before decomposition\n")
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    #print(sess.run(T))
    print("\n")

# Creating e2
    b_pseoudoInv = np.linalg.pinv(b)
    project = np.dot(b, b_pseoudoInv)
    e2 = np.dot(project, e1)
    
    g = tf.Variable(tf.random_uniform([N]))
    #print('\n g is \n')
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    print(sess.run(g))


    T2 = T_new = tf.zeros([len(e2),len(e2),len(e2)], tf.float32)
    for i in range(0):
        T1 = decom(T,g)
        T_new += T-T1
        T = T1
        T2 = T_new - Tfirst
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        n = tf.norm(T2)
        print(sess.run(n))
        print("\n")
        
    norm = tf.nn.l2_normalize(T_new, 0, epsilon = 1e-12, name = None)
    print("sum of decomposition:\n")
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    print(sess.run(T_new))
    print("\n")    


        

if __name__ == '__main__':
    main()


