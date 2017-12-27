import numpy as np

def initialize_weights_and_bias(d_in, num_classes):

    W = np.random.randn(d_in, num_classes)
    b = np.random.randn(1, num_classes)

    return W, b

def make_one_hot(y, num_classes):

    y_one_hot = np.zeros((len(y), num_classes))
    y_one_hot[np.arange(len(y)), y] = 1

    return y_one_hot


def forward_pass(X, W, b):

    y = np.matmul(X, W) + np.repeat(b, np.shape(X)[0], axis=0)

    return y

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.repeat(np.expand_dims(np.sum(np.exp(x), axis=1), axis=1), np.shape(x)[1], axis=1)

def relu(x):

    return np.maximum(x, 0)

def compute_cost(y, y_pred):

    y_pred = softmax(y_pred)
    cross_entropy = np.sum(np.multiply(y, np.log(y_pred)))

    return cross_entropy

def softmax_regression(X_train, Y_train, X_test, Y_test):

    num_training_samples = np.shape(X_train)[0]
    num_test_samples = np.shape(X_test)[0]
    d_in = np.shape(X_train)[-1]
    num_classes = len(np.unique(np.concatenate((Y_train, Y_test))))

    W, b = initialize_weights_and_bias(d_in, num_classes)

    Y_train = make_one_hot(Y_train, num_classes)
    Y_test = make_one_hot(Y_test, num_classes)

    learning_rate = 0.001
    for t in range(100):

        Y_train_pred = forward_pass(X_train, W, b)
        cost = compute_cost(Y_train, Y_train_pred)
        print(t, cost)

        # Backprop to compute gradients of w1 and w2 with respect to loss



if __name__ == '__main__':
    # Create random input and output data
    X_train = np.random.randn(998, 128)
    Y_train = np.random.randint(low=0, high=10, size=(998,))
    X_test = np.random.randn(203, 128)
    Y_test = np.random.randint(low=0, high=10, size=(203,))
    softmax_regression(X_train, Y_train, X_test, Y_test)







