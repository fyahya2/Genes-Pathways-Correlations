import tensorflow as tf
import numpy as np
import math
import json
#import matplotlib
#import matplotlib.pyplot as plt
#%matplotlib inline
#matplotlib.style.use('ggplot')


#--------------------------------------------------

# Converting to tensor

def my_func(arg):
    print('\nWe entered to function---> my_func')
    
    arg = tf.convert_to_tensor(arg)
    return arg

#--------------------------------------------------

# Computing significances of ith subtensor

def fraction(eigenexprssionsi, M):
    print('\nWe entered to function---> fraction')
    
 #   ai_fractions = ai_sSquare / np.sum(ai_sSquare)
    print(eigenexprssionsi)
    arr = eigenexprssionsi
    lambdaSum = np.sum(arr)
    aiFrac = np.zeros(M)
    for i in range(M):
        aiFrac[i] = arr[i] / lambdaSum

    return aiFrac
#--------------------------------------------------

# Calculating entropy

def ent(data):

    print('\nWe entered to function---> ent')
    
    entropy = 0
    for i in range(0,len(data)):
        entropy = entropy + (data[i] * np.log2(data[i]))
            
    entropy = -1/np.log2(12) *entropy
    
    return entropy

#---------------------------------------------

# Main algorithm to decompose the tensor

def decom(T,N):

    print('\nWe entered to function---> decom')
    
    g = tf.Variable(tf.random_normal([N]))

    L = 1
    N_iter = 5
 # This is for L in power method
    for i in range(L):
        X = tf.Variable(tf.random_uniform([N,N]))
        norm = tf.norm(X)
        X = X / norm
        for j in range(N_iter): # This is for N in power method
            #X = tf.einsum('abf,cdl,flk,k,bd->ac',T,T,T,g,X)
            X = tf.einsum('abf,flk,ak->bl',T,T,X)
            norm = tf.norm(X)
            X = X / norm
        E = tf.self_adjoint_eig(X, name = None)
        
        print('\nWe just finished eigenvalue function')
        
        g = E[1][:,N-1]
           
    print('\n End of power method\Here we have created Eigenvector')

    for j in range(N_iter): # This is for N in power method
        g = tf.einsum('klm,m,l->k',T,g,g)
        norm = tf.norm(g)
        g = g / norm
        
    eigen = tf.einsum('klm,m,l,k',T,g,g,g)
    T = tf.einsum('i,j,k->ijk',g,g,g)
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    return (g, sess.run(eigen),sess.run(g),T)

#--------------------------------------------------

#Individual tensor decomposition and significances

def Indv_decom(T, N, iteration):

    print('\nWe entered to function---> Indv_decom')
    
    #T = tf.slice(T, [0,0,0], [N,N,N] )

    Tsum = tf.zeros([N,N,N])
    Tfirst = T

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)

    print("\n norm of original Tensor:\n")
    n = tf.norm(T)
    print(sess.run(n))
    
  #  T = T/n
    
    Comp = np.zeros((iteration,N))
    Eigen =[]
    Indivcounter = 0
    n0 = n
    n1 = tf.constant(0.00)
    
    for i in range(iteration):
        vec, eig,comp,T = decom(T,N)
        
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)

        print(sess.run(n0))
        print(sess.run(n1))
       # if eig<0 or sess.run(n0)<sess.run(n1):
       #     break
        
        Eigen.append(eig)
        Comp[i] = comp
        T = eig * T
        Tsum += T
        T = Tfirst - Tsum 
        n0 = n
        n = tf.norm(T)
        n1 = n
        print("\n")
        print(sess.run(n))
        Indivcounter = Indivcounter + 1

 # list of eigenvalues
 
    Eigen = tf.stack(Eigen)
    
 # Calculating significance of subTensors of T

    TFrac = fraction(sess.run(Eigen),Indivcounter)
    print("\nExpression correlations for most significant subTensors of T(%)\n")
    print(TFrac*100)
    print('\n')
    T_entropy = ent(TFrac)
    print("Entropy of Tensor T\n")
    print(T_entropy)

    return (T_entropy, TFrac*100, Eigen, Comp, Indivcounter)



#--------------------------------------------------             # Main function

def main():

 # Number of iterations for both individual and overall tensor
 
    iteration1 = 18
    iteration2 = 12
    iteration3 = 12
    iteration4 = 8
    N = 27

#--------------------------------------------------
    #Individual tensor decomposition

 # Tensor 1 Decomposition

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    
    print('-------------------------------\n')
    print('\t\t\t********* Tensor 1 decomposition *********')
    with open('/Users/farzaneh/Desktop/New_file/T1.txt') as f:
    	data = json.load(f)

    T1 = my_func(data)
 #   print(T1)
    print('Tensor 1 is created')

    entropy, aiFrac, Eigen1, Comp1, Indivcounter1 = Indv_decom(T1, N, iteration1)
    
    with open('/Users/farzaneh/Desktop/New_file/entropy1.txt','w') as fout:
        fout.write(str(entropy))
    
    with open('/Users/farzaneh/Desktop/New_file/Indiv_Sig1.txt','w') as fout:
        fout.write(str(aiFrac))
    
    with open('/Users/farzaneh/Desktop/New_file/eigenvector1.txt','w') as fout:
        fout.write(str(Comp1))
    
    with open('/Users/farzaneh/Desktop/New_file/eigenvalue1.txt','w') as fout:
        fout.write(str(sess.run(Eigen1)))

#--------------------------------------------------
    
 # Tensor 2 Decomposition
 
    tf.reset_default_graph()

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    
    print('-------------------------------\n')
    print('\t\t\t********* Tensor 2 decomposition *********')
    with open('/Users/farzaneh/Desktop/New_file/T2.txt') as f:
    	data = json.load(f)

    T2 = my_func(data)
    print('Tensor 2 is created')

    entropy, aiFrac, Eigen2, Comp2, Indivcounter2 = Indv_decom(T2, N, iteration2)
    
    with open('/Users/farzaneh/Desktop/New_file/entropy2.txt','w') as fout:
        fout.write(str(entropy))
    
    with open('/Users/farzaneh/Desktop/New_file/Indiv_Sig2.txt','w') as fout:
        fout.write(str(aiFrac))
    
    with open('/Users/farzaneh/Desktop/New_file/eigenvector2.txt','w') as fout:
        fout.write(str(Comp2))
    
    with open('/Users/farzaneh/Desktop/New_file/eigenvalue2.txt','w') as fout:
        fout.write(str(sess.run(Eigen2)))

#--------------------------------------------------
    
 # Tensor 3 Decomposition
 
    tf.reset_default_graph()

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    
    print('-------------------------------\n')
    print('\t\t\t********* Tensor 3 decomposition *********')
    with open('/Users/farzaneh/Desktop/New_file/T3.txt') as f:
        data = json.load(f)
    T3 = my_func(data)
    print('Tensor 3 is created') 

    entropy, aiFrac, Eigen3, Comp3, Indivcounter3 = Indv_decom(T3, N, iteration3)

    with open('/Users/farzaneh/Desktop/New_file/entropy3.txt','w') as fout:
        fout.write(str(entropy))
    
    with open('/Users/farzaneh/Desktop/New_file/Indiv_Sig3.txt','w') as fout:
            fout.write(str(aiFrac))

    with open('/Users/farzaneh/Desktop/New_file/eigenvector3.txt','w') as fout:
        fout.write(str(Comp3))
    
    with open('/Users/farzaneh/Desktop/New_file/eigenvalue3.txt','w') as fout:
        fout.write(str(sess.run(Eigen3)))

        
#--------------------------------------------------
    
 # Tensor 4 Decomposition
 
    tf.reset_default_graph()

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    
    print('-------------------------------\n')
    print('\t\t\t********* Tensor 4 decomposition *********')
    with open('/Users/farzaneh/Desktop/New_file/T4.txt') as f:
        data = json.load(f)
    T4 = my_func(data)
    print('Tensor 4 is created')
    entropy, aiFrac, Eigen4, Comp4, Indivcounter4 = Indv_decom(T4, N, iteration4)

    with open('/Users/farzaneh/Desktop/New_file/entropy4.txt','w') as fout:
        fout.write(str(entropy))
    
    with open('/Users/farzaneh/Desktop/New_file/Indiv_Sig4.txt','w') as fout:
        fout.write(str(aiFrac))

    with open('/Users/farzaneh/Desktop/New_file/eigenvector4.txt','w') as fout:
        fout.write(str(Comp4))
    
    with open('/Users/farzaneh/Desktop/New_file/eigenvalue4.txt','w') as fout:
        fout.write(str(sess.run(Eigen4)))


 
    
if __name__ == '__main__':
    main()
