import numpy as np
import pandas as pd
import scipy as sc
import math

#---------------------------------------------------

#Finding relevant data of genes in datasets

def relData(interSec, genes, M):

    print(len(interSec))
    print(len(genes))
    for j in range(len(genes)):
        if genes[j] in interSec:
            with open('Cell_Cycle_Binding.txt','r') as infile:
                with open("Intersection_b1.txt", "a") as outfile:
                    for i, line in enumerate(infile):
                        if i ==j:
                            outfile.write("\t".join(line.split()) + "\n")
                            
#---------------------------------------------------

def main():

    #Creating list of gene names of e1
        
    f = open('GeneNames_Cell_Cycle.txt','r')
    sig_genes = f.readlines()
    print("Length of signal genes of e1: ")
    print(len(sig_genes))
    print('\n')

    #Creating list of gene names of b1
    
    f = open("GeneNames_Cycle_Bin.txt",'r')
    b1_genes = f.readlines()
    print("Length of signal genes of b1: ")
    print(len(b1_genes))
    print('\n')

    #Creating list of gene names of b2
        
    f = open("GeneNames_Dev_Bin.txt",'r')
    b2_genes = f.readlines()
    print("Length of signal genes of b2: ")
    print(len(b2_genes))
    print('\n')

    #Creating list of gene names of b3
        
    f = open("GeneNames_Biosyn_Bin.txt",'r')
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
    print('\n')
    
#---------------------------------------------------

    #Finding relevant data of intersection(e1, b1, b2, and b3) in e1
    M=12
    relData(interSec, b1_genes, M)

    

if __name__ == '__main__':
    main()    
