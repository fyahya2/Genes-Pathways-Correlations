import tensorflow as tf
import numpy as np
import math
import json


# Converting to tensor

def my_func(arg):
    print('\nWe entered to function---> my_func')
    
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg


def main():

 # Number of iterations for both individual and overall tensor


    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    
#--------------------------------------------------
 # Tensor 1 
 #   T1 = tf.placeholder(tf.float32, [])
    print('-------------------------------\n')
    print('\t\t\t********* Tensor 1 *********')
    with open('/Users/farzaneh/Desktop/GeneTensor/T4_50.txt') as f:
        data1 = json.load(f)
    T1 = my_func(data1)

    print('Tensor 1 was created')
#--------------------------------------------------

 #   T_all = my_func(T50_all)
    n = sess.run(T1)

    with open('/Users/farzaneh/Desktop/GeneTensor/test.txt', 'w') as outfile:
    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in n:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

    print('\ndone')
 #   with sess.as_default():
 #       sess.run(T50_all).eval().tofile('/Users/farzaneh/Desktop/GeneTensor/T50_all.txt',sep='')
        

if __name__ == '__main__':
    main()