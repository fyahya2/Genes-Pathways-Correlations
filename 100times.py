import numpy as np

e1 = np.loadtxt( "/Users/farzaneh/Desktop/result27-300/eigenvalue1.txt")
e2 = np.loadtxt( "/Users/farzaneh/Desktop/result27-300/eigenvalue2.txt")
e3 = np.loadtxt( "/Users/farzaneh/Desktop/result27-300/eigenvalue3.txt")
e4 = np.loadtxt( "/Users/farzaneh/Desktop/result27-300/eigenvalue4.txt")
e = np.loadtxt( "/Users/farzaneh/Desktop/result27-300/eigenvalue_all.txt")
e1 = np.reshape(e1,(20,1))
e2 = np.reshape(e2,(12,1))
e3 = np.reshape(e3,(12,1))
e4 = np.reshape(e4,(8,1))
e = np.reshape(e,(8,1))
s1 = sum(abs(e1))
s2 = sum(abs(e2))
s3 = sum(abs(e3))
s4 = sum(abs(e4))
s = s1 + s2 + s3 + s4

f1 = np.loadtxt( "/Users/farzaneh/Desktop/result27-300/lamdas1.txt")
f2 = np.loadtxt( "/Users/farzaneh/Desktop/result27-300/lamdas2.txt")
f3 = np.loadtxt( "/Users/farzaneh/Desktop/result27-300/lamdas3.txt")
f4 = np.loadtxt( "/Users/farzaneh/Desktop/result27-300/lamdas4.txt")

f1 = f1/s
f2 = f2/s
f3 = f3/s
f4 = f4/s
f1 = np.reshape(f1,(816,1))
print(f1[0])
print(f1[1])
print(f1[15])
print(f1[120])
print('\n')

f2 = np.reshape(f2,(220,1))
print(f2[0])
print(f2[1])
print(f2[9])
print(f2[55])
print('\n')

f3 = np.reshape(f3,(220,1))
print(f3[0])
print(f3[1])
print(f3[9])
print(f3[55])
print('\n')

f4 = np.reshape(f4,(56,1))
print(f4[0])
print(f4[1])
print(f4[5])
print(f4[28])
print('\n')


