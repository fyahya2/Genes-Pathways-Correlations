import tensorflow as tf

def decom(T,N):
    
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    
    Max = tf.zeros([0])
    vec = tf.Variable(tf.random_uniform([N]))
    Max1 = 0
    L = 1
    N_iter = 5
 # This is for L in power method
    for i in range(L):
        g = tf.Variable(tf.random_uniform([N]))
        norm = tf.norm(g)
        g = g / norm
        #N_iter = sess.run(tf.cast(tf.norm(T), tf.int32))
        for j in range(N_iter): # This is for N in power method
            g = tf.einsum('ack,k,c->a',T,g,g)
            norm = tf.norm(g)
            g = g / norm

        m = tf.einsum('zyo,o,y,z',T,g,g,g)

        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        if sess.run(tf.less(Max,m)):
            Max = m
            vec = g        
    g = vec
    
    for j in range(N_iter): # This is for N in power method
        g = tf.einsum('klm,m,l->k',T,g,g)
        norm = tf.norm(g)
        g = g / norm
        
    eigen = tf.einsum('klm,m,l,k',T,g,g,g)
    print("\n eigenvalue: \n")
    print(sess.run(eigen))
    print("\n top eigenvector\n")
    print(sess.run(g))        
    T = tf.einsum('i,j,k->ijk',g,g,g)   
    T = eigen * T
    return T

def main():

    a1 = 14
    a2 = 1
    a3 = 28
    vec1 = tf.constant([1, 1, 1], dtype = tf.float32)
    norm = tf.norm(vec1)
    vec1 = vec1 / norm
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)    
    print(sess.run(vec1))    
    
    sub_T1 = tf.einsum('k,m,l->kml',vec1,vec1,vec1)
    sub_T1 = a1 * sub_T1

    vec2 = tf.constant([1, -1, 0], dtype = tf.float32)
    norm = tf.norm(vec2)
    vec2 = vec2 / norm     
    sub_T2 = tf.einsum('k,m,l->kml',vec2,vec2,vec2)
    sub_T2 = a2 * sub_T2

    vec3 = tf.constant([0, 1, -1], dtype = tf.float32)
    norm = tf.norm(vec3)
    vec3 = vec3 / norm     
    sub_T3 = tf.einsum('k,m,l->kml',vec3,vec3,vec3)
    sub_T3 = a3 * sub_T3

    T_main = T = sub_T1 + sub_T2 + sub_T3
    
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)    
    print(sess.run(T))
    
    print(sess.run(tf.norm(T_main)))
    
    N = 3

    for i in range(6):
        T = decom(T, N)
        T_main -= T
        T = T_main
    n = tf.norm(T)
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)     
    print(sess.run(n))
        

if __name__ == '__main__':
    main()
