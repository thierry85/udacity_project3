# Load pickled data
import pickle
import pandas as pd
import numpy as  np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

# TODO: Fill this in based on where you saved the training and testing data

training_file = "./traffic-signs-data/train.p"
validation_file="./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
df= pd.Series(y_train)
n_classes = len(pd.unique(df))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)

print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#df=df.value_counts()
#df.plot.bar()
#plt.show()

# preproces  data
##########
from sklearn.utils import shuffle
img_brut=X_train[0];
X_train= np.sum(X_train/3, axis=3, keepdims=True)
X_valid= np.sum(X_valid/3, axis=3, keepdims=True)
X_test= np.sum(X_test/3, axis=3, keepdims=True)
img_gray=X_train[0];
# normalisation
X_train = (X_train -128)/128
X_valid = (X_valid -128)/128
X_test = (X_test -128)/128
img_norm=X_train[0];
X_train, y_train = shuffle(X_train, y_train)
# Plot the result
#f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
#f.tight_layout()
#ax1.imshow(img_brut)
#ax1.set_title('Original Image', fontsize=40)
#ax2.imshow(img_gray)
#ax2.set_title('gray image', fontsize=40)
#ax3.imshow(img_norm)
#ax3.set_title('normalised image', fontsize=40)
# on screen
#plt.show()

print("end preprocessing normalisation and gray")

#setup tensorflow
##########
import tensorflow as tf

EPOCHS = 50
BATCH_SIZE = 500
#implement lenet-5
##########
from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    #  Activation.
    conv1 = tf.nn.relu(conv1)

    #  Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    #  Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    #  Activation.
    fc1    = tf.nn.relu(fc1)
    # dropout
    fc1=tf.nn.dropout(fc1,keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    #  Activation.
    fc2    = tf.nn.relu(fc2)
    fc2=tf.nn.dropout(fc2,keep_prob)

    #  Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

# features and labels
##########
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

#training pipeline
##########
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


#model evaluation
##########
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# train the model
##########
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  num_examples = len(X_train)
  
  print("Training...")
  print()
  for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.6})
    validation_accuracy = evaluate(X_valid, y_valid)
    print("EPOCH {} ...".format(i+1))
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    print()
      
  saver.save(sess, './lenet')
  print("Model saved")

#evaluate the model
 ##########
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.import_meta_graph('./lenet.meta')
  saver.restore(sess, "./lenet")

  test_accuracy = evaluate(X_test, y_test)
  print("Test Accuracy = {:.3f}".format(test_accuracy))

# test on other images
#####################
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

x_img=[]
for file in sorted(os.listdir("./Other_signs/")):
  image=mpimg.imread('./Other_signs/' + file)
  mid=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  mid=mid.reshape(32,32,1)
  mid = (mid - 128)/128
  x_img.append(mid)
x_label=[3,34,25,14,13]

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.import_meta_graph('./lenet.meta')
  saver.restore(sess, "./lenet")
  my_accuracy = evaluate(x_img, x_label)
  print("Test Set Accuracy = {:.3f}".format(my_accuracy))



softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=3)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: x_img, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: x_img, keep_prob: 1.0})
    print(my_top_k)
