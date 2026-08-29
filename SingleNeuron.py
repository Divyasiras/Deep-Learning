import numpy as np

#Step 1 : define input feature ie X
#                  [X1, X2, X3]
input = np.array([2.0,3.0,4.0])
print("X :",input)

#Step 2 : define weights ie W
#                  [W1, W2, W3]
weights = np.array([0.5,0.3,0.2])
print("W",weights)

#step 3 : define bias ie b
bias = 1.0
print("b :",bias)

#step 4 : calculate weighted sum e Z
# z = x1w1 + x2w2 + x3w3 + b
# z = (2.0*0.5) + (3.0*0.3) + (4.0*0.2) + 1.0

z = np.dot(input,weights)+ bias
print("Z :",z)

# Step 5 : Actication function (ReLU)
def ReLU(x):
    return max(0,x)

# Step 6 : final output
Y = ReLU(z)
print("Y :",Y)


