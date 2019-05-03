import tensorflow as tf

def main():


    vec = tf.constant( [ 0.01317366, 0.0018154, 0.39742094, 0.02713084,0.07800478, 0.12891553
,0.18407579,0.30535498,0.26726487,0.35407224,0.1996318, 0.16095366
, 0.25309378, 0.08976985,0.04498831,0.09635755, 0.08362927,0.03453688
, 0.03794894,0.05290147,0.02668313,0.35369658,0.27367142, 0.13227697
, 0.23727077,0.14699246,0.1918273 ], dtype = tf.float32)
    sub_T = tf.einsum('k,m,l->kml',vec,vec,vec)
    
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    n = sess.run(T)

    with open('/Users/farzaneh/Desktop/New_file/T1_1.txt', 'w') as outfile:
    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in n:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

if __name__ == '__main__':
    main()
