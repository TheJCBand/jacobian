import numpy as np
import time

N = 500  # Input size
H = 100  # Hidden layer size
M = 10   # Output size

w1 = np.random.randn(N, H)  # first affine layer weights
b1 = np.random.randn(H)     # first affine layer bias

w2 = np.random.randn(H, M)  # second affine layer weights
b2 = np.random.randn(M)     # second affine layer bias

import tensorflow as tf
from tensorflow.keras.layers import Dense

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

model = tf.keras.Sequential()
model.add(Dense(H, activation='relu', use_bias=True, input_dim=N))
model.add(Dense(M, activation='softmax', use_bias=True, input_dim=M))
model.get_layer(index=0).set_weights([w1, b1])
model.get_layer(index=1).set_weights([w2, b2])

def jacobian_tensorflow(x):
    jacobian_matrix = []
    for m in range(M):
        # We iterate over the M elements of the output vector
        grad_func = tf.gradients(model.output[:, m], model.input)
        gradients = sess.run(grad_func, feed_dict={model.input:x.reshape((1, x.size))})
        jacobian_matrix.append(gradients[0][0,:])

    return np.array(jacobian_matrix)

def is_jacobian_correct(jacobian_fn, ffpass_fn):
    """ Check of the Jacobian using numerical differentiation
    """
    x = np.random.random((N,))
    epsilon = 1e-5

    """ Check a few columns at random
    """
    for idx in np.random.choice(N, 5, replace=False):
        x2 = x.copy()
        x2[idx] += epsilon

        num_jacobian = (ffpass_fn(x2) - ffpass_fn(x)) / epsilon
        computed_jacobian = jacobian_fn(x)

        if not all(abs(computed_jacobian[:, idx] - num_jacobian) < 1e-3):
            return False

    return True

def ffpass_tf(x):
    """ The feedforward function of our neural net
    """
    xr = x.reshape((1, x.size))
    return model.predict(xr)[0]

print(is_jacobian_correct(jacobian_tensorflow, ffpass_tf))

x0 = np.random.random((N,))
tic = time.time()
jacobian_tf = jacobian_tensorflow(x0)
tac = time.time()

print('It took %.3f s. to compute the Jacobian matrix' % (tac-tic))

import autograd.numpy as anp

def ffpass_anp(x):
    a1 = anp.dot(x, w1) + b1   # affine
    a1 = anp.maximum(0, a1)    # ReLU
    a2 = anp.dot(a1, w2) + b2  # affine

    exps = anp.exp(a2 - anp.max(a2))  # softmax
    out = exps / exps.sum()
    return out

out_anp = ffpass_anp(x0)
out_keras = ffpass_tf(x0)

print(np.allclose(out_anp, out_keras, 1e-4))

from autograd import jacobian

def jacobian_autograd(x):
    return jacobian(ffpass_anp)(x)

print(is_jacobian_correct(jacobian_autograd, ffpass_anp))

tic = time.time()
jacobian_autograd(x0)
tac = time.time()

print('It took %.3f s. to compute the Jacobian matrix' % (tac-tic))
