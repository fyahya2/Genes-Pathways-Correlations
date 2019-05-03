import tensorflow as tf
import numpy as np
import math
import json


# Converting to tensor

def my_func(arg):
    print('\nWe entered to function---> my_func')
    
    arg = tf.convert_to_tensor(arg, dtype=tf.float16)
    return arg


def main():

 # Number of iterations for both individual and overall tensor


    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
   
 # Tensor 1 Decomposition
 
 #  T_12 = tf.placeholder(tf.float16, [])
    print('-------------------------------\n')
    print('\t\t\t********* Tensor 12 *********')
    with open('/Users/farzaneh/Desktop/GeneTensor/T_12.txt') as f:
        data = f.read().split()
    T = [[[0 for i in range(514)] for j in range(514)] for k in range(514)]
    print(data[0])
    z=0
    for i in range(0,514):
        for j in range(0,514):
            for k in range(0,514):
                T[i][j][k]=data[z]
                z=z+1
 #   print(T1)
    with open('/Users/farzaneh/Desktop/GeneTensor/T12_edited.txt','w') as fout:
        fout.write(str(T))

    with open('/Users/farzaneh/Desktop/GeneTensor/T12_edited.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('/Users/farzaneh/Desktop/GeneTensor/T12_edited.txt', 'w') as fout:
        for line in data:
            fout.write(line.replace("'", " "))

    print('Tensor 12 was edited')

    with open('/Users/farzaneh/Desktop/GeneTensor/T12_edited.txt') as f:
        data = json.load(f)
    T1 = my_func(data)
#--------------------------------------------------
    
 # Tensor 2 Decomposition
 #   T_34 = tf.placeholder(tf.float16, [])
    print('-------------------------------\n')
    print('\t\t\t********* Tensor 34 *********')
    with open('/Users/farzaneh/Desktop/GeneTensor/T_34.txt') as f:
        data = f.read().split()

    print(data[0])
    z=0
    for i in range(0,514):
        for j in range(0,514):
            for k in range(0,514):
                T[i][j][k]=data[z]
                z=z+1
#   print(T1)
    with open('/Users/farzaneh/Desktop/GeneTensor/T34_edited.txt','w') as fout:
        fout.write(str(T))
    
    with open('/Users/farzaneh/Desktop/GeneTensor/T34_edited.txt', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('/Users/farzaneh/Desktop/GeneTensor/T34_edited.txt', 'w') as fout:
        for line in data:
            fout.write(line.replace("'", " "))

    print('Tensor 34 was edited')

    with open('/Users/farzaneh/Desktop/GeneTensor/T34_edited.txt') as f:
        data = json.load(f)

    T2 = my_func(data)
    print('\nEnter to summation for overall tensor')
#--------------------------------------------------

 # Creating overal Tensor(order 4 tensor): K*N*N*N K=4

    T_all = T1 + T2
#   T_all = my_func(T_all)
        #    with open('/Users/farzaneh/Desktop/GeneTensor/T_all.txt','a') as fout:
#       fout.writeline(str(sess.run(T_all)))

    n = sess.run(T_all)

    with open('/Users/farzaneh/Desktop/GeneTensor/T_all.txt', 'w') as outfile:
    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in n:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
            np.savetxt(outfile, data_slice, fmt='%-7.2f')
            
 #   with sess.as_default():
 #       T_all.eval().tofile('/Users/farzaneh/Desktop/GeneTensor/T_all.txt',sep='')

if __name__ == '__main__':
    main()
