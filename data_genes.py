import numpy as np
import scipy as sc
import math

def main():


    #Finding relevant data of intersection(e1, b1, b2, and b3) in e1
    M=12
 #   relData(interSec, sig_genes, M)
    with open('/Users/farzaneh/Desktop/New_file/Removed_None_b4.txt','r') as infile:
        for line in infile:
            with open("/Users/farzaneh/Desktop/New_file/Intersection_genes.txt",'r') as infile2:
                for line2 in infile2:
                    if line.split()[0:1] ==line2.split()[0:1]:
                        with open('/Users/farzaneh/Desktop/New_file/b4_data.txt', 'a') as outfile:
                            outfile.write("\t".join(line.split()) + "\n")
    

if __name__ == '__main__':
    main()    
