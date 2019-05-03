import numpy as np
import pandas as pd
import scipy as sc
import math
from sklearn.preprocessing import Imputer
from fancyimpute import IterativeSVD, BiScaler, KNN, NuclearNormMinimization, SoftImpute

#---------------------------------------------

#Writing on the file

def Clear_Cell_Cycle(line, outfile):
    outfile.write("\t".join(line.split()[7:30]) + "\n")
    return

#---------------------------------------------


#Finding relevant data of genes in datasets

def relData(interSec, sig, genes, M):

    b1_e1 = np.zeros((1588,M))
    k=0
    
    for j in range(0,len(genes)):
        if genes[j] in interSec:
            b1_e1[k] = sig[j,:]
            k=k+1           
    newData = b1_e1
    
    print("Dimension of intersectinon matrix of e1 and b1 of b1 ")
    print(newData.shape)
    print('\n')
    return newData

#------------------------------------------------

def main():

    #Enter the raw data file (main signal matrix e1)
    
    with open("Removed_None_b2.txt","r") as infile:
        with open("b2.txt", "w") as outfile:
            for line in infile:
                Clear_Cell_Cycle(line, outfile);

    #Keeping name of the genes and features of e1 (Cell_Cycle_Expresion)
        
    with open('Removed_None_b2.txt', "r") as infile:
        with open("GeneNames_b2.txt", "w") as outfile:
            for line in infile:
                outfile.write("\t".join(line.split()[0:1]) + "\n")

    #Creating list of signal genes
        
    f = open("GeneNames_b2.txt",'r')
    sig_genes = f.readlines()
    print("Length of signal genes of b2: ")
    print(len(sig_genes))
    print('\n')
               
    #Changing NULL to 0
        
    with open("b2.txt", 'r') as fin2:
        data = fin2.read().splitlines(True)
    with open("b2.txt", 'w') as fout2:
        for line in data:
            fout2.write(line.replace('Null', '0'))



if __name__ == '__main__':
    main()
            
