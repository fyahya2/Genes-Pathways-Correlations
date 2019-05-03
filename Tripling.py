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
    
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

#--------------------------------------------------

#Computing significances of ith subtensor

def fraction(eigenexprssionsi, M, N):
    print('\nWe entered to function---> fraction')
    
 #   ai_fractions = ai_sSquare / np.sum(ai_sSquare)
    if M == 1 :
        arr = [eigenexprssionsi]
    else:
        arr = eigenexprssionsi
    lambdaSum = np.sum(arr)
    print(arr)
    aiFrac = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            aiFrac[i][j] = arr[i][j] / lambdaSum

    return aiFrac
#--------------------------------------------------

#Calculating entropy

def ent(data):

    print('\nWe entered to function---> ent')
    
    entropy = 0
    for i in range(0,len(data)):
        entropy = entropy + (data[i] * np.log2(data[i]))
            
    entropy = -1/np.log2(12) *entropy
    
    return entropy

#--------------------------------------------------
# Main algorithm to decompose the tensor

def decom(T,N):

    print('\nWe entered to function---> decom')
    
    g = tf.Variable(tf.random_normal([N]))

    L_iter = 1
    N_iter = 300
    
 # This is for L in power method
    for i in range(L_iter):
        X = tf.Variable(tf.random_normal([N,N]))        
        norm = tf.norm(X)
        X = X / norm
        for j in range(N_iter): # This is for N in power method
            X = tf.einsum('abf,cdl,flk,k,bd->ac',T,T,T,g,X)
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

#---------------------------------------------

def Indv_HOT(Indivcounter, allcounter, eVec, Ten_list, Eigen, N):

    print('\nWe entered to function---> Indv_HOT')
    
 # Checking number of eigen of individuals and subtensors of overall network
    
    print('Eigenvalues of individual:\n')
    print(Indivcounter)
    print('\n Subtensors of overall:\n')
    print(allcounter)    
    if allcounter < Indivcounter:
        iteration = allcounter
    else:
        iteration = Indivcounter
        
    
 # Creating all combinations of components N*N-1*N-2/6
 
    T_comb = []
    count = 0
    for i in range(iteration):
        for j in range(i+1,iteration):
            for k in range(j+1,iteration):
                T = tf.einsum('m,n,o',eVec[i],eVec[j],eVec[k])
                T = T + tf.einsum('m,n,o',eVec[i],eVec[k],eVec[j])
                T = T + tf.einsum('m,n,o',eVec[j],eVec[i],eVec[k])
                T = T + tf.einsum('m,n,o',eVec[j],eVec[k],eVec[i])
                T = T + tf.einsum('m,n,o',eVec[k],eVec[i],eVec[j])
                T = T + tf.einsum('m,n,o',eVec[k],eVec[j],eVec[i])
                T_comb.append(T)
                count = count + 1
                
    print("\n number of tripling:\n")
    print(count)
    print('\n eigen\n')
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    print(sess.run(Eigen))
    #print('\n tensors\n')
    #print(sess.run(Ten_list))
    T_comb = tf.stack(T_comb, axis =1)

#--------------------------------------------------
    
 # Finding the same size slices of eigenvalues and subtensors for Ti
 # to calculate the correlations of Trippleing
 
    Eigen = tf.slice(Eigen, [0], [iteration])
    Ten_list = tf.slice(Ten_list, [0,0,0,0], [iteration,N,N,N] )
    
    T1_remain = T - tf.einsum('i,ijkl->jkl',Eigen,Ten_list)
    reshapeT1 = tf.reshape(T1_remain,[N*N*N,1])
    #print(sess.run(reshapeT1))
    #print('\n')
    reshapeT = tf.reshape(T_comb,[N*N*N,count])
    #print(sess.run(reshapeT))
    triple_lamdas = tf.matrix_solve_ls(reshapeT,reshapeT1)

    print("Tripling\n")
    print(sess.run(triple_lamdas))

    return (triple_lamdas)

#--------------------------------------------------             # Main function

def main():

 # Number of iterations for both individual and overall tensor
 
    iteration = 4
    N = 27

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
#--------------------------------------------------            #Individual tensor decomposition results

 # Tensor 1
 
    with open('/Users/farzaneh/Desktop/result27-300/eigenvalue1.txt','r') as fout:
        data = json.load(fout)
    Eigen1 = my_func(data)

#--------------------------------------------------
    
 # Tensor 2

    with open('/Users/farzaneh/Desktop/result27-300/eigenvalue2.txt') as fout:
        data = json.load(fout)
    Eigen2 = my_func(data)

#--------------------------------------------------
    
 # Tensor 3
    
    with open('/Users/farzaneh/Desktop/result27-300/eigenvalue3.txt') as fout:
        data = json.load(fout)
    Eigen3 = my_func(data)
        
#--------------------------------------------------
    
 # Tensor 4

    with open('/Users/farzaneh/Desktop/result27-300/eigenvalue4.txt') as fout:
        data = json.load(fout)
    Eigen4 = my_func(data)
        
#--------------------------------------------------

 # Creating overal Tensor(order 4 tensor): K*N*N*N K=4

    data = tf.placeholder(tf.float32, [])
    print('-------------------------------\n')
    print('\t\t\t********* The whole Tensor decomposition *********')
    with open('/Users/farzaneh/Desktop/result27-300/T_all.txt') as f:
        data = json.load(f)
        
    T_a = T_all = tf.slice(data, [0,0,0], [N,N,N] )

    Indivcounter1 = Indivcounter2 = Indivcounter3 = Indivcounter4 = 3

 # Overall Tensor decomposition
 
    T1_remain = T = Tsum = tf.zeros([N,N,N])

    #print("\n overall Tesnor before decomposition\n")
 #   init_op = tf.global_variables_initializer()
 #   sess = tf.Session()
 #   sess.run(init_op)
    #print(sess.run(T_all))
    print("\n norm of original overall Tensor:\n")
    n = tf.norm(T_all)
    print(sess.run(n))    

    Eigen = []
    Ten_list = []
    eVec = []
    allcounter = 0
    n0 = n
    n1 = tf.constant(0.0)
    
    for j in range(iteration):
        
        vec, eigen, comp,T_all = decom(T_all,N)

        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        
        with tf.Graph().as_default():
            sess.run(init_op)        
            x = sess.run(n0)
            y = sess.run(n1)
        print(x)
        print(y)
        #if eigen < 0 or x < y:
        #    break

        Eigen.append(eigen)
        Ten_list.append(T_all)
        eVec.append(vec)
        Tsum += eigen * T_all
        T_all = T_a - Tsum 
        n0 = n
        n = tf.norm(T_all)
        n1 = n
        print("\n")
        print(sess.run(n))
        allcounter = allcounter + 1

    Eigen = tf.stack(Eigen)
    
 # list of subtensors of overall tensor
 
    Ten_list = tf.stack(Ten_list)

 # list of eigenvectors of overall tensor
 
    eVec = tf.stack(eVec)
    print(sess.run(eVec))

    with open('/Users/farzaneh/Desktop/result27-300/eigenvector_all.txt','w') as fout:
        fout.write(str(sess.run(eVec)))
    
    with open('/Users/farzaneh/Desktop/result27-300/eigenvalue_all.txt','w') as fout:
        fout.write(str(sess.run(Eigen)))
    
#--------------------------------------------------
    
 # Calculating significance of subTensors of overal tensor
    M = 1
    TallFrac = fraction(sess.run(Eigen),M, allcounter)
    print("Expression correlations for most significant subTensors of overall T(%)")
    print(TallFrac*100)
    print('\n')
    Tall_entropy = ent(TallFrac)
    print("Entropy of Tensor T1")
    print(Tall_entropy)
    print("fraction of first eigenvalue of overall T")
    print(TallFrac[0])
    print('\n')

    with open('/Users/farzaneh/Desktop/result27-300/Indiv_Sig_all.txt','w') as fout:
        fout.write(str(TallFrac))
    
#--------------------------------------------------

    # Higher order tensor formulation for individual tensors

    triple_lamdas1= Indv_HOT(Indivcounter1, allcounter, eVec, Ten_list, Eigen1, N)
    with open('/Users/farzaneh/Desktop/result27-300/lamdas1.txt','w') as fout:
        fout.write(str(sess.run(triple_lamdas1)))
    fout.close()
    
    triple_lamdas2= Indv_HOT(Indivcounter2, allcounter, eVec, Ten_list, Eigen2, N)
    with open('/Users/farzaneh/Desktop/result27-300/lamdas2.txt','w') as fout:
        fout.write(str(sess.run(triple_lamdas2)))
    fout.close()
    
    triple_lamdas3 = Indv_HOT(Indivcounter3, allcounter, eVec, Ten_list, Eigen3, N)
    with open('/Users/farzaneh/Desktop/result27-300/lamdas3.txt','w') as fout:
        fout.write(str(sess.run(triple_lamdas3)))
    fout.close()
    
    triple_lamdas4 = Indv_HOT(Indivcounter4, allcounter, eVec, Ten_list, Eigen4, N)
    with open('/Users/farzaneh/Desktop/result27-300/lamdas4.txt','w') as fout:
        fout.write(str(sess.run(triple_lamdas4)))
    fout.close()
   
#--------------------------------------------------

 # Significance of mth subtensor in the kth tensor

    if allcounter < Indivcounter1:
        iteration = allcounter
    else:
        iteration = Indivcounter1

    if Indivcounter2 < iteration:
        iteration = Indivcounter2

    if Indivcounter3 < iteration:
        iteration = Indivcounter3

    if Indivcounter4 < iteration:
        iteration = Indivcounter4

    Eigen1 = tf.slice(Eigen1, [0], [iteration])
    Eigen2 = tf.slice(Eigen2, [0], [iteration])
    Eigen3 = tf.slice(Eigen3, [0], [iteration])
    Eigen4 = tf.slice(Eigen4, [0], [iteration])

    eigAll = []
    eigAll.append(Eigen1)
    eigAll.append(Eigen2)
    eigAll.append(Eigen3)
    eigAll.append(Eigen4)
    eigAll = tf.stack(eigAll)

    m = 4
    eigFrac = fraction(sess.run(eigAll),m,iteration)
    print("Expression correlations for mth subtensor in kth tensor (%)")
    print(eigFrac *100)
    print('\n')
    eigFrac_entropy = ent(eigFrac )
    print("Entropy of Tensor all")
    print(Tall_entropy)
    print("fraction of first eigenvalue of overall T")
    print(eigFrac[0])
    print('\n')
    
    with open('eigFrac.text','a') as fout:
        fout.write(str(eigFrac))
    fout.close()


    # Removing extra '[' from triple_lamdas file before sending to fraction

    # lamdas1.txt

    with open('/Users/farzaneh/Desktop/result27-300/lamdas1.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('/Users/farzaneh/Desktop/result27-300/lamdas1.txt', 'w') as fout:
        for line in data:
            fout.write(line.replace('[[', '['))
    with open('/Users/farzaneh/Desktop/result27-300/lamdas1.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('/Users/farzaneh/Desktop/result27-300/lamdas1.txt', 'w') as fout:
        for line in data:
            fout.write(line.replace(']]', ']'))

    # lamdas2.txt

    with open('/Users/farzaneh/Desktop/result27-300/lamdas2.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('/Users/farzaneh/Desktop/result27-300/lamdas2.txt', 'w') as fout:
        for line in data:
            fout.write(line.replace('[[', '['))
    with open('/Users/farzaneh/Desktop/result27-300/lamdas2.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('/Users/farzaneh/Desktop/result27-300/lamdas2.txt', 'w') as fout:
        for line in data:
            fout.write(line.replace(']]', ']'))

    # lamdas3.txt

    with open('/Users/farzaneh/Desktop/result27-300/lamdas3.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('/Users/farzaneh/Desktop/result27-300/lamdas3.txt', 'w') as fout:
        for line in data:
            fout.write(line.replace('[[', '['))
    with open('/Users/farzaneh/Desktop/result27-300/lamdas3.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('/Users/farzaneh/Desktop/result27-300/lamdas3.txt', 'w') as fout:
        for line in data:
            fout.write(line.replace(']]', ']'))

    # lamdas4.txt

    with open('/Users/farzaneh/Desktop/result27-300/lamdas4.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('/Users/farzaneh/Desktop/result27-300/lamdas4.txt', 'w') as fout:
        for line in data:
            fout.write(line.replace('[[', '['))
    with open('/Users/farzaneh/Desktop/result27-300/lamdas4.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('/Users/farzaneh/Desktop/result27-300/lamdas4.txt', 'w') as fout:
        for line in data:
            fout.write(line.replace(']]', ']'))

    print('triple_lamdas file was edited')
    TrippleAll = []
    TrippleAll.append(triple_lamdas1[0][0])
    TrippleAll.append(triple_lamdas2[0][0])
    TrippleAll.append(triple_lamdas3[0][0])
    TrippleAll.append(triple_lamdas4[0][0])
    TrippleAll = tf.stack(TrippleAll)

    m = 1
    TrippleFrac = fraction(sess.run(TrippleAll), m, iteration)
    print("\nExpression correlations for mth subtensor in kth tensor (%)")
    print(eigFrac *100)
    TrippleFrac_entropy = ent(TrippleFrac )
    print("Entropy of Tensor T1")
    print(TrippleFrac_entropy)
    print("fraction of first eigenvalue of overall T")
    print(TrippleFrac[0])
    print('\n')
    
    with open('TrippleFrac.text','w') as fout:
        fout.write(str(TrippleFrac))
    fout.close()

#--------------------------------------------------

    tf.reset_default_graph()

    with open('/Users/farzaneh/Desktop/result27-300/T1.txt','r') as fout:
        data = json.load(fout)
    Comp1 = my_func(data)

    with open('/Users/farzaneh/Desktop/result27-300/T2.txt','r') as fout:
        data = json.load(fout)
    Comp2 = my_func(data)

    with open('/Users/farzaneh/Desktop/result27-300/T3.txt','r') as fout:
        data = json.load(fout)
    Comp3 = my_func(data)

# Analyzing correlation T1 and T2

    for i in range(len(Comp1)):
        for j in range(len(Comp2)):
             with open('/Users/farzaneh/Desktop/result27-300/corr12.txt','a') as fout:
                 fout.write(str(np.correlate(Comp1[i],Comp2[j])) + ',')
        with open('/Users/farzaneh/Desktop/result27-300/corr12.txt','a') as fout:
            fout.write('\n')
    fout.close()

# Analyzing correlation T1 and T3

    for i in range(len(Comp1)):
        for j in range(len(Comp3)):
            with open('/Users/farzaneh/Desktop/result27-300/corr13.txt','a') as fout:
                fout.write(str(np.correlate(Comp1[i],Comp3[j])) + ',')
        with open('/Users/farzaneh/Desktop/result27-300/corr13.txt','a') as fout:
            fout.write('\n')
    fout.close()

# Analyzing correlation T2 and T3

    for i in range(len(Comp2)):
        for j in range(len(Comp3)):
            with open('/Users/farzaneh/Desktop/result27-300/corr23.txt','a') as fout:
                fout.write(str(np.correlate(Comp2[i],Comp3[j])) + ',')
        with open('/Users/farzaneh/Desktop/result27-300/corr23.txt','a') as fout:
            fout.write('\n')
    fout.close()

    #plt.scatter(Comp1, Comp2)
    #plt.show()

 
    
if __name__ == '__main__':
    main()
