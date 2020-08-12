import numpy as np


def check_array_size(array):
    """ Check if array has more than 1 dimension """
    array_shape = array.shape
    if len(array_shape) < 2:
        return 0
    if array_shape[0] < 2:
        return 0
    if array_shape[1] < 2:
        return 0
    return 1


def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    """

    norm_predictions = predictions - np.amax(predictions, axis=1)[:, None]
    exp_array = np.exp(norm_predictions)
    return exp_array/np.sum(exp_array, axis=1)[:, None]


def cross_entropy_loss(probs, target_index, batch_size=1, num_classes=1):
    """
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
      batch_size is a number of batches in probs array
    Returns:
      loss: single value
    """

    mask_array = np.zeros((batch_size, num_classes), dtype=int)
    # print(f'Probs\n{probs}')
    # print(f'Target Index\n{target_index}')
    for i in range(mask_array.shape[0]):
        mask_array[i, target_index[i]] = 1
    # print(f'Mask Array\n{mask_array}')
    ce_loss = -np.sum(mask_array * np.log(probs), axis=1)
    # print(f'CE_Loss\n{ce_loss}')
    return np.average(ce_loss)


def softmax_with_cross_entropy(predictions, target_index, batch_size=1, num_classes=1):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """

    probs = softmax(predictions)
    # print(f'Probs\n{probs}')
    loss = cross_entropy_loss(probs, target_index, batch_size, num_classes)
    # print(f'CE Loss\n{loss}')
    target_array = np.zeros((batch_size, num_classes), dtype=int)
    # print(f'Target index\n{target_index}')

    for i in range(target_array.shape[0]):
        target_array[i, target_index[i]] = 1
    # print(f'Target array\n{target_array}')

    dprediction = probs - target_array
    # print(f'Dprediction\n{dprediction}')
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            raise Exception("Not implemented!")

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
