import numpy as np
from random import shuffle
import pdb

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero(10,3073)

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):#(0,49000)
    scores = W.dot(X[:, i]) #(10,3073).dot((3073,))=(10,)
    correct_class_score = scores[y[i]] #0.21718258108235677
    for j in xrange(num_classes):#(0,10)
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        # compute gradients (one inner and one outer sum)
        # sum over j != y_i
        dW[y[i],:] -= X[:,j].T#X(3073,49000)
        # sums each contribution of xi
        dW[j,:] += X[:,i].T 
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  D = X.shape[0]
  num_class = W.shape[0]
  num_train = X.shape[1]
  scores = W.dot(X) #shape CxN(10,49000)
  # construct correct_score shape Dx1??? (49000,) Nx1
  # array([-0.14733042, -0.19433695,  0.09016157, ...,  0.07690333,
  #        0.20338156, -0.2007116 ])
  correct_scores = scores[y, np.arange(num_train)]

  mat = scores - correct_scores + 1 #(10,49000)
  # according for the j=y_i term, we shouldnt count
  # since w_j == w_{y_j}, substracting 1 for j = y_i
  mat[y, np.arange(num_train)] = 0

  thresh = np.maximum(np.zeros((num_class,num_train)), mat) #(10,49000)

  loss = np.sum(thresh)
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  # binarize into intergers
  binary = thresh#(10,49000)
  binary[thresh>0] = 1
  # perform the two operations simultaneously
  # (1) for all j: dW[j:] = sum_{i,} X[:,i].T
  # (2) for all i: dW[y[i],:] = sum_{j != y_i,}  -X[:,i].T
  col_sum = np.sum(binary, axis=0)#(49000)
  binary[y,range(num_train)] = -col_sum[range(num_train)]
  dW = np.dot(binary, X.T)

  dW /= num_train

  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
