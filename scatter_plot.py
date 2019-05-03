import matplotlib.pyplot as plt
 
# x-axis values
x = [1,2,3]
     
# plotting points as a scatter plot
y = [ 2.70702869e+01,   8.33584834e-01,   1.02118086e+00]
plt.plot('x', y, color= "green", marker= "*", linestyle='dashed')

# plotting points as a scatter plot
y= [2.36343682e+01,   8.07173201e-03,   6.23927335e-02]
plt.plot('x', y, color= "blue",marker= "*", linestyle='dashed')

# plotting points as a scatter plot
y= [2.32529804e+01,   1.59924268e-02,   1.60945972e-02]
plt.plot('x', y, color= "red",marker= "*", linestyle='dashed')

# plotting points as a scatter plot
y= [2.40605339e+01,   3.00749725e-03,   2.15108754e-02]
plt.plot('x', y, color= "black",marker= "*", linestyle='dashed')

# x-axis label
plt.xlabel('Tensors')
# frequency label
plt.ylabel('components')
# showing legend
plt.legend()
 
# function to show the plot
plt.show()
