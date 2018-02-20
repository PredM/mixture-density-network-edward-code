# Code refers to this tutorial: https://github.com/supergitacc/mixture-density-network-edward-code/new/master

import tensorflow as tf
import numpy as np
import seaborn as sns
import edward as ed
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def build_toy_dataset(N):
  y_data = np.random.uniform(-10.5, 10.5, N)
  r_data = np.random.normal(size=N)  # random noise
  x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
  x_data = x_data.reshape((N, 1))
  return train_test_split(x_data, y_data, random_state=42)

N = 5000  # number of data points
D = 1  # number of features

X_train, X_test, y_train, y_test = build_toy_dataset(N)
print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))

sns.regplot(X_train, y_train, fit_reg=False)

X_ph = tf.placeholder(tf.float32, [None, D])
y_ph = tf.placeholder(tf.float32, [None])

from edward.models import Categorical, Mixture, Normal

def neural_network(X):
  """loc, scale, logits = NN(x; theta)"""
  # 2 hidden layers with 15 hidden units
  net = tf.layers.dense(X, 15, activation=tf.nn.relu)
  net = tf.layers.dense(net, 15, activation=tf.nn.relu)
  locs = tf.layers.dense(net, K, activation=None)
  scales = tf.layers.dense(net, K, activation=tf.exp)
  logits = tf.layers.dense(net, K, activation=None)
  return locs, scales, logits

K = 20  # number of mixture components

locs, scales, logits = neural_network(X_ph)
cat = Categorical(logits=logits)
components = [Normal(loc=loc, scale=scale) for loc, scale
              in zip(tf.unstack(tf.transpose(locs)),
                     tf.unstack(tf.transpose(scales)))]
y = Mixture(cat=cat, components=components, value=tf.zeros_like(y_ph))

inference = ed.MAP(data={y: y_ph})

optimizer = tf.train.AdamOptimizer(5e-3)
inference.initialize(optimizer=optimizer, var_list=tf.trainable_variables())

sess = ed.get_session()
tf.global_variables_initializer().run()

n_epoch = 1000
train_loss = np.zeros(n_epoch)
test_loss = np.zeros(n_epoch)
for i in range(n_epoch):
  info_dict = inference.update(feed_dict={X_ph: X_train, y_ph: y_train})
  train_loss[i] = info_dict['loss']
  test_loss[i] = sess.run(inference.loss, feed_dict={X_ph: X_test, y_ph: y_test})
  inference.print_progress(info_dict)
  
pred_weights, pred_means, pred_std = sess.run(
    [tf.nn.softmax(logits), locs, scales], feed_dict={X_ph: X_test})
	
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 3.5))
plt.plot(np.arange(n_epoch), -test_loss / len(X_test), label='Test')
plt.plot(np.arange(n_epoch), -train_loss / len(X_train), label='Train')
plt.legend(fontsize=20)
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Log-likelihood', fontsize=15)
plt.show()
