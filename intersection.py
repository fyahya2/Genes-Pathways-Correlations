import numpy as np
import pandas as pd
import scipy as sc
import math
from sklearn.preprocessing import Imputer
from fancyimpute import IterativeSVD, BiScaler, KNN, NuclearNormMinimization, SoftImpute
import tensorflow as tf

#---------------------------------------------------

# Function for creating a tensor

def TensorProj(M):

    le = len(M.T)
    N = len(M)
    print('we entereted to TensorProj func\n')
  #  T = [[[0 for i in range(N)] for j in range(N)] for k in range(N)]
 #   T = tf.zeros([len(e1),len(e1),len(e1)], tf.float32)
 #   print('empty Twas created\n')
    with open('T4.txt','a') as fout:
        fout.write('[')
    for i in range(0,N):
        with open('T4.txt','a') as fout:
            fout.write('[')
        for j in range(0,N):
            with open('T4.txt','a') as fout:
                fout.write('[')
            for k in range(0,N):
                x=0
                for l in range(0,le):
                    x += M[i,l] * M[j,l]* M[k,l]
                with open('T4.txt','a') as fout:
                    fout.write(str(x) + ',')
 #                   T[i][j][k] += M[i,l] * M[j,l]* M[k,l]
            with open('T4.txt','a') as fout:
                fout.write(']')
        with open('T4.txt','a') as fout:
            fout.write(']')
        #print(i)
    with open('T4.txt','a') as fout:
        fout.write(']')
 #   fout.close()

#---------------------------------------------------
 
#Finding relevant data of genes in datasets
 
def relData(interSec, sig, genes, M):

    b_e = np.zeros((len(interSec),M))
    k=0

    print(len(interSec))
    print(len(genes))
    for j in range(len(genes)):
        if genes[j] in interSec:
            b_e[k] = sig[j,:]
            k=k+1           
    newData = b_e
    
    return newData

#---------------------------------------------------

def main():

    #Creating matrix e1
            
    sig_matrix = np.loadtxt('e1.txt')
    print("Dimension of raw e1: ")
    print(sig_matrix.shape)
    print('\n')
    
    #Filling missing data in e1
    
    sig_matrix[sig_matrix == 0] = np.NaN
    X_incomplete = sig_matrix

    # Use SVD
    X_filled = IterativeSVD().complete(X_incomplete)

    sig_matrix = X_filled
#---------------------------------------------------
    #Creating matrix basis signal b1
        
    sig_basis1 = np.loadtxt('b1.txt')
    print("Dimension of b1(Cell Cycle Binding): ")
    print(sig_basis1.shape)
    print('\n')
#---------------------------------------------------
    #Creating matrix basis signal b2
        
    sig_basis2 = np.loadtxt('b2.txt')
    print("Dimension of b2: ")
    print(sig_basis2.shape)
    print('\n')
#---------------------------------------------------
    #Creating matrix basis signal b3
        
    sig_basis3 = np.loadtxt('b3.txt')
    print("Dimension of b3: ")
    print(sig_basis3.shape)
    print('\n')
#---------------------------------------------------

    #Creating list of gene names of e1
        
    f = open('GeneNames_e1.txt','r')
    sig_genes = f.readlines()
    print("Length of signal genes of e1: ")
    print(len(sig_genes))
    print('\n')

    #Creating list of gene names of b1
    
    f = open("GeneNames_b1.txt",'r')
    b1_genes = f.readlines()
    print("Length of signal genes of b1: ")
    print(len(b1_genes))
    print('\n')

    #Creating list of gene names of b2
        
    f = open("GeneNames_b2.txt",'r')
    b2_genes = f.readlines()
    print("Length of signal genes of b2: ")
    print(len(b2_genes))
    print('\n')

    #Creating list of gene names of b3
        
    f = open("GeneNames_b3.txt",'r')
    b3_genes = f.readlines()
    print("Length of signal genes of b3: ")
    print(len(b3_genes))
    print('\n')

#---------------------------------------------------
    
    #Find intersection of e1, b1, b2, and b3
    
    interSec = set(sig_genes).intersection(b1_genes)
    print(len(interSec))    
    interSec = set(interSec).intersection(b2_genes)
    print(len(interSec))
    interSec = set(interSec).intersection(b3_genes)
    print("Length of intersection\n")
    print(len(interSec))
    print(interSec)

    #Finding relevant data of intersection(e1, b1, b2, and b3) in e1
    M=18
    print('\n') 
    sig_matrix = relData(interSec, sig_matrix, sig_genes, M)
    print("Length of intersectinon in a1 ")
    print(len(sig_matrix))
    print('\n')    

    #Finding relevant data of intersection(e1, b1, b2, and b3) in b1
    M=12
    sig_basis1 = relData(interSec, sig_basis1, b1_genes, M)
    print("Length of intersectinon in b1 ")
    print(len(sig_basis1))
    print('\n')    

    #Finding relevant data of intersection(e1, b1, b2, and b3) in b2
    M=12
    sig_basis2 = relData(interSec, sig_basis2, b2_genes, M)
    print("Length of intersectinon in b2 ")    
    print(len(sig_basis2))
    print('\n')

    #Finding relevant data of intersection(e1, b1, b2, and b3) in b3
    M=8
    sig_basis3 = relData(interSec, sig_basis3, b3_genes, M)
    print("Length of intersectinon in b3 ")
    print(len(sig_basis3))
    print('\n')

#---------------------------------------------------
    
    #Devide signal matrices by mean to convert signals to DNA binding

    sig_npArray = np.array(sig_matrix)
    sig_mean = np.mean(sig_npArray, axis=1)
    print("Mean of e1 for gene measurments:")
    print(sig_mean)
    for i in range(27):
        sig_matrix[i, :] = sig_matrix[i, :] / sig_mean[i]
    print("\n")
    
    sig_npArray = np.array(sig_basis1)
    basis1_mean = np.mean(sig_npArray, axis=1)
    print("Mean of b1 for gene measurments:")
    print(basis1_mean)
    for i in range(27):
        sig_basis1[i, :] = sig_basis1[i, :] / basis1_mean[i]
    print("\n")

    sig_npArray = np.array(sig_basis2)
    basis2_mean = np.mean(sig_npArray, axis=1)
    print("Mean of b2 for gene measurments:")
    print(basis1_mean)
    for i in range(27):
        sig_basis2[i, :] = sig_basis2[i, :] / basis2_mean[i]
    print("\n")


    sig_npArray = np.array(sig_basis3)
    basis3_mean = np.mean(sig_npArray, axis=1)
    print("Mean of b3 for gene measurments:")
    print(basis1_mean)
    for i in range(27):
        sig_basis3[i, :] = sig_basis3[i, :] / basis3_mean[i]
    print("\n")

#---------------------------------------------------

    # Creating tesnors

    e1 = sig_matrix
 #   TensorProj(e1)
    
 # Creating e2
 
 #   b1_pseoudoInv = np.linalg.pinv(sig_basis1)
 #   project = np.dot(sig_basis1, b1_pseoudoInv)
 #   e2 = np.dot(project, e1)
 #   TensorProj(e2)

 # Creating e3
 
 #   b2_pseoudoInv = np.linalg.pinv(sig_basis2)
 #   project = np.dot(sig_basis2, b2_pseoudoInv)
 #   e3 = np.dot(project, e1)
 #   TensorProj(e3)

 # Creating e4
 
    b3_pseoudoInv = np.linalg.pinv(sig_basis3)
    project = np.dot(sig_basis3, b3_pseoudoInv)
    e4 = np.dot(project, e1)
    TensorProj(e4)
 
#---------------------------------------------------


if __name__ == '__main__':
    main()    
