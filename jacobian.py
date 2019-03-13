import tensorflow as tf
import numpy as np 

# Function for defining simple MLP network
def mlp(net,nUnits,outputUnits,activation=tf.nn.relu,outputActivation=None,trainable=True,name=None):
    for layer in range(len(nUnits)):
        net=tf.layers.dense(net,nUnits[layer],activation=activation,trainable=True)
    return tf.layers.dense(net,outputUnits,activation=outputActivation,name=name)

xDim = 10
yDim = 3
x = tf.placeholder(tf.float32, shape=(None, xDim))
y = mlp(x,[64,64],yDim)


def defineJacobianOps(net,yDim,x):
    jacobianOps = []
    for j in range(yDim):
        jacobianOps.append(tf.gradients(net[:,j],x))
    return jacobianOps

jacobianOps = defineJacobianOps(y,yDim,x)
            
sess = tf.Session()
sess.run(tf.initializers.global_variables())

x0 = np.random.random((xDim,))
jacobian = sess.run(jacobianOps, feed_dict={x:x0.reshape((1,xDim))})
jacobian = np.squeeze(np.array(jacobian))
print(jacobian)
print(np.shape(jacobian))
