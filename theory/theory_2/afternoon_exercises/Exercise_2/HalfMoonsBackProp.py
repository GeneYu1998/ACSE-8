"""
Author: Lukas Mosser, modified by O. Dubrule, in particular the equations used for the backward 
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from   sklearn.datasets     import make_moons
from   sklearn.metrics      import accuracy_score

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return True

def make_train_test(batch_size, batch_num, test_size, noise):
    """
    Makes a half-moon train-test dataset with fixed batch size, number and noise level
    """
    X_train, y_train = make_moons(n_samples=batch_size*batch_num, noise=noise)
    y_train          = y_train.reshape(batch_num, batch_size, 1)
    X_train          = X_train.reshape(batch_num, batch_size, 2)
    X_test, y_test   = make_moons(n_samples=test_size, noise=noise)
    y_test           = y_test.reshape(test_size, 1)
    return X_train, y_train, X_test, y_test
"""
Training a single-hidden-layer neural network
"""
set_seed(42)
epochs     = 1000 
"""
Setting up some hyperparameters
"""
batch_size = 10000 #Size of a single batch
batch_num  = 1    #Use full batch training
test_size  = 1000  #Examples in test setset
lr         = 1.
D, H, M    = 2, 10, 1 #Define input size, Size of Hidden Layer, Output size
"""
This is where momentum is chosen. Set it to some value between 0 and 1 for momentum option
"""
momentum = 0.8
"""
Create two-moons + noise
"""
X_train, y_train, X_test, y_test = make_train_test(batch_size, batch_num, test_size, noise=0.3)
"""
Convert to torch tensors, single batch
"""
X = torch.from_numpy(X_train).float()[0] 
y = torch.from_numpy(y_train).float()[0] 
"""
Convert to torch tensors, already single batch
"""
X_test = torch.from_numpy(X_test).float() 
y_test = torch.from_numpy(y_test).float() 
"""
Plot training and test dataset
"""
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(X[:, 0], X[:, 1], c=y[:, 0])
ax[0].set_title('Training set') 
ax[0].set_xlabel('x1') 
ax[0].set_ylabel('x2') 
ax[0].legend()
ax[1].scatter(X_test[:, 0], X_test[:, 1],c=y_test[:, 0])
ax[1].set_title('Test set') 
ax[1].set_xlabel('x1') 
ax[1].set_ylabel('x2') 
ax[1].legend()
"""
Define Sigmoid Activation Functions and Derivatives
"""
sigmoid  = lambda x: 1./(1+torch.exp(-x)) 
dSigmoid = lambda x: x*(1-x) 
"""
Initialize neural network weights as random numbers and biases at zero
"""
W1, W2 = torch.randn((D, H)), torch.randn((H, M)) 
b1, b2 = torch.zeros((H))   , torch.zeros((M)) 
"""
Define the momentum parameters in case momentum is used
"""
if momentum is not None:
    vt_1_W1 = 0.
    vt_1_W2 = 0.
    vt_1_b1 = 0.
    vt_1_b2 = 0.
"""
Start Training
"""
for i in range(epochs):
    """
    N is number of Training examples
    """
    N = X.size(0)
    """
    Forward Passes Layers 1 and 2
    """
    z1 = torch.matmul(X, W1)+b1 
    a1 = sigmoid(z1) # a1(N,H) contains at each line the output of the first layer for this datapoint
    z2 = torch.matmul(a1, W2)+b2 
    a2 = sigmoid(z2) #a2(N,M) contains at each line the output of the second layer for this datapoint
    """
    Backward Pass Layer 2
    """
    dL_da2 = (a2-y)/(a2*(1-a2)) #Modified by OD: Compute Error on Output
    da2_dz2= dSigmoid(a2) #Compute derivative of activation function (Sigmoid)
    dL_dW2 = torch.matmul(torch.transpose(a1, 0, 1), dL_da2*da2_dz2) #Compute gradient w.r.t. weights in layer 2       
    dL_db2 = (dL_da2*da2_dz2).sum(0) #Compute gradient w.r.t. bias in layer 2, sums over all N examples
    """
    Backward Pass Layer 1
    """
    dL_da1 = torch.matmul(dL_da2*da2_dz2, torch.transpose(W2, 0, 1)) #Modified by OD: Compute Error on Output of Layer 1
    da1_dz1= dSigmoid(a1) #Compute derivative of activation function (Sigmoid)
    dL_dW1 = torch.matmul(torch.transpose(X, 0, 1), dL_da1*da1_dz1) #Compute gradient w.r.t. weights in layer 2
    dL_db1 = (dL_da1*da1_dz1).sum(0)  #Compute gradient w.r.t. bias in layer 1, sums over all N examples
    """
    Sensitivity of loss function with relation to input
    """
    dL_dX = torch.matmul(dL_da1*da1_dz1, torch.transpose(W1, 0, 1)) #Modified by OD: Compute gradient w.r.t. input X
    """
    Gradient descent with momentum
    """
    if momentum is not None:
        """
        The value of momentum controls the weighted average between current and previous step.
        New = Momentum*Old+(1-Momentum)*∂L/(∂W_1 )
        See good explanation in https://engmrk.com/gradient-descent-with-momentum/
        """
        vt_W1 = momentum*vt_1_W1+(1-momentum)*(lr*dL_dW1)/N #Modified by OD: Momentum step for layer 1 weights
        W1 = W1 - vt_W1 #Take a step in momentum weighted direction on layer 1 weights
        vt_1_W1 = vt_W1
        
        vt_W2 = momentum*vt_1_W2+(1-momentum)*(lr*dL_dW2)/N
        W2 = W2 - vt_W2  #Modified by OD: Take a step in momentum weighted direction on layer 2 weights
        vt_1_W2 = vt_W2       

        vt_b1 = momentum*vt_1_b1+(1-momentum)*(lr*dL_db1)/N
        b1 = b1 - vt_b1 #Modified by OD: Take a step in momentum weighted direction on layer 1 bias
        vt_1_b1 = vt_b1     
        
        vt_b2 = momentum*vt_1_b2+(1-momentum)*(lr*dL_db2)/N
        b2 = b2 - vt_b2  #Modified by OD: Take a step in momentum weighted direction on layer 2 bias
        vt_1_b2 = vt_b2   
     
    else: 
        """
        Standard Gradient Descent Approach
        """
        
        W1 = W1 - (lr*dL_dW1)/N #Take a step in gradient direction on layer 1 weights
        b1 = b1 - (lr*dL_db1)/N #Take a step in gradient direction on layer 1 bias

        W2 = W2 - (lr*dL_dW2)/N #Take a step in gradient direction on layer 2 weights
        b2 = b2 - (lr*dL_db2)/N #Take a step in gradient direction on layer 2 bias
             
    """
    Compute average training cross-entropy loss
    """
    train_loss = -(y*torch.log(a2)+(1-y)*torch.log(1-a2)).mean(0) 
    
    if i % 100 == 0:
        print("Training Loss in epoch "+str(i)+": %1.2f" % train_loss.item())
        print("Training accuracy in epoch "+str(i)+": %1.2f" % accuracy_score(np.where(a2[:, 0].numpy()>0.5, 1, 0), y),"\n")
"""
Forward pass on test dataset (both layers) and calculate cross-entropy loss on test dataset
"""
z1_t      = torch.matmul(X_test, W1)+b1 
a1        = sigmoid(z1_t) 
z2        = torch.matmul(a1, W2)+b2 
a_test    = sigmoid(z2)
test_loss = -(y_test*torch.log(a_test)+(1-y_test)*torch.log(1-a_test)).mean(0)

print("End of Training -> Testing Phase: ")
print("Train Loss: %1.2f" % train_loss.item(), ", Test Loss: %1.2f" % test_loss.item())
print("Training accuracy in epoch "+str(i)+": %1.2f" % accuracy_score(np.where(a2[:, 0].numpy()>0.5, 1, 0), y))
print("Test accuracy in epoch "+str(i)+": %1.2f" % accuracy_score(np.where(a_test[:, 0].numpy()>0.5, 1, 0), y_test))
"""
Plot the results of neural network on training and test dataset
"""    
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(X[:, 0], X[:, 1]          , c=np.where(a2[:, 0]    .numpy()>0.5, 1, 0))
ax[0].set_title('Neural Network Forecasts on Training Set') 
ax[0].set_xlabel('x1') 
ax[0].set_ylabel('x2') 
ax[0].legend()
ax[1].scatter(X_test[:, 0], X_test[:, 1], c=np.where(a_test[:, 0].numpy()>0.5, 1, 0))
ax[1].set_title('Neural Network Forecasts on Test Set') 
ax[1].set_xlabel('x1') 
ax[1].set_ylabel('x2') 
ax[1].legend()
"""
Show the distribution of weights and biases
"""
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
"""
Plot Histograms of W1 (blue) and W2 (orange)
"""
ax[0].hist(W1.numpy().flatten(),label='Hidden Layer')
ax[0].hist(W2.numpy().flatten(),label='Output layer')
ax[0].set_title('Histograms of weights') 
ax[0].set_xlabel('Weights') 
ax[0].set_ylabel('Number of occurrences') 
ax[0].legend()
"""
Plot Histograms of b1 (blue) and b2 (orange)
"""
ax[1].hist(b1.numpy().flatten(),label='Hidden Layer')
ax[1].hist(b2.numpy().flatten(),label='Output Layer')
ax[1].set_title('Histograms of biasses') 
ax[1].set_xlabel('Biasses') 
ax[1].set_ylabel('Number of occurrences') 
ax[1].legend()
"""
Show training sets and use sensitivities as labels, thus highlighting the points
of the cluster that have the largest impact on the loss function 
"""
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.scatter(X[:, 0], X[:, 1], c=dL_dX[:, 0].abs())
plt.xlabel('X')
plt.ylabel('Y')
plt.title ('Sensitivity of loss function to each point of the Training Set')