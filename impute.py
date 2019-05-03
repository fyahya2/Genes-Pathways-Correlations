from fancyimpute import IterativeSVD
import numpy as np

def imputeData(X_incomplete):

    X = np.zeros((4153,18))
    

    # X_incomplete has missing data which is represented with NaN values
    X_filled = IterativeSVD().complete(X_incomplete)
    print('test1')
    return

def main():
    
    # X is the complete data matrix
    # X_incomplete has the same values as X except a subset have been replace with NaN

    #Changing NULL to NaN
        
    with open("Cell_Cycle_Expresion.txt", 'r') as fin2:
        data = fin2.read().splitlines(True)
    with open("Cell_Cycle_NaN.txt", 'w') as fout2:
        for line in data:
            fout2.write(line.replace('Null', '0'))
    print('test')
    f = open("Cell_Cycle_NaN.txt", 'r')
 #   X_incomplete = f.read()
 #   X_incomplete = [map(int, line.split(',')) for line in f]
    X_incomplete = np.loadtxt('Cell_Cycle_NaN.txt')
    X_incomplete[X_incomplete == 0] = np.NaN
    
    imputeData(X_incomplete)
    print('test')
if __name__ == '__main__':
    main()
