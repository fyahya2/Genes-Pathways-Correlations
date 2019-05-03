import numpy as np
import scipy as sc
import math

def main():


    #Finding name of the genes in 120 correlations
    with open('/Users/farzaneh/Desktop/120Corr_T124.txt','r') as infile:
        for line in infile:
            with open("/Users/farzaneh/Desktop/27genes.txt",'r') as infile2:
                for i, line2 in enumerate(infile2):
                    j = list(map(int, line.split()))[0]
                    k = list(map(int, line.split()))[1]
                    l = list(map(int, line.split()))[2]
                    #print(j)
                    if i == j:
                        #print(i)
                        with open('/Users/farzaneh/Desktop/T124_Column1.txt', 'a') as outfile:
                            outfile.write("\t".join(line2.split()) + "\n")    

                    if i == k:
                        #print(i)
                        with open('/Users/farzaneh/Desktop/T124_Column2.txt', 'a') as outfile:
                            outfile.write("\t".join(line2.split()) + "\n")

                    if i == l:
                        #print(i)
                        with open('/Users/farzaneh/Desktop/T124_Column3.txt', 'a') as outfile:
                            outfile.write("\t".join(line2.split()) + "\n")                             
if __name__ == '__main__':
    main()    
