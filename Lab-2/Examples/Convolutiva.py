import tensorflow as tf
import os
from Helper_functions import plot_images,plot_example_errors,print_confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

##################################### Dados ##########################################

data = input_data.read_data_sets("data/MNIST/", one_hot=True)

################################## Parametros ########################################

learning_rate = 0.001
training_iters = 40000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

################################ Modelo (Grafo) ######################################

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32, name="keep_prob")

#----------------------------- Arquitetura -------------------------------------------

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(features, weights, biases, keep_prob):

    input_layer = tf.reshape(features, shape=[-1, 28, 28, 1])

    #----------------------------------------- conv1 ------------------------------------------
    
    conv1 = conv2d(input_layer, weights['w_c1'], biases['b_c1'])
    
    #---------------------------------------- pool1 -------------------------------------------
    
    pool1 = maxpool2d(conv1, k=2)
    
    #---------------------------------------- conv2 -------------------------------------------
    
    conv2 = conv2d(pool1, weights['w_c2'], biases['b_c2'])
    
    #---------------------------------------- pool2 -------------------------------------------
    
    pool2 = maxpool2d(conv2, k=2)
    
    #--------------------------------------- local3 -------------------------------------------
    
    flat = tf.reshape(pool2, [-1, weights['w_d1'].get_shape().as_list()[0]])
    local3 = tf.add(tf.matmul(flat, weights['w_d1']), biases['b_d1'])
    local3 = tf.nn.relu(local3)
    
    #--------------------------------------- local3 -------------------------------------------
    
    dropout4 = tf.nn.dropout(local3, keep_prob)
    
    #--------------------------------------- softmax -------------------------------------------
    
    logits = tf.add(tf.matmul(dropout4, weights['w_out']), biases['b_out'])
    
    return logits

#----------------------------- Variaveis (pesos e bias) -------------------------------

weights = {
'w_c1': tf.Variable(tf.random_normal([5, 5, 1, 12])),
'w_c2': tf.Variable(tf.random_normal([5, 5, 12, 12])),
'w_d1': tf.Variable(tf.random_normal([7*7*12, 128])),
'w_out': tf.Variable(tf.random_normal([128, n_classes]))
}

biases = {
'b_c1': tf.Variable(tf.random_normal([12])),
'b_c2': tf.Variable(tf.random_normal([12])),
'b_d1': tf.Variable(tf.random_normal([128])),
'b_out': tf.Variable(tf.random_normal([n_classes]))
}

#---------------------------------- Treinamento ----------------------------------------

logits = conv_net(x, weights, biases, keep_prob)

prediction = tf.nn.softmax(logits)
classe = tf.argmax(logits,1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


################################## Executar Grafo ######################################

#----------------------------- Alimentador Placeholder ---------------------------------

def feed_data(train=True):
    
    x_batch, y_batch = data.train.next_batch(batch_size)
    
    if train:
        return {x: x_batch, y: y_batch, keep_prob: 0.75}
    else:
        return {x: x_batch, y: y_batch, keep_prob: 1}

#-------------------------------------- Session  ---------------------------------------

session = tf.Session()
session.run(tf.global_variables_initializer())

step = 1

while step * batch_size < training_iters:
    
    session.run(optimizer, feed_dict=feed_data())

    if step % display_step == 0:
        
        loss, acc = session.run([cost, accuracy], feed_dict=feed_data(False))
               
        print("Iter %i, Minibatch Loss= %f, Training Accuracy= %f" % (step*batch_size , loss , acc))
        
    step += 1

print("Otimização Finalizada!")

# Funções de ajuda para avaliar posteriormente no tutorial

feed_dict_test={x: data.test.images, y: data.test.labels, keep_prob: 1.}

def print_accuracy():
    session.run(accuracy , feed_dict=feed_dict_test)
    print("Acuracia: {0:.1%}".format(acc) )
    
def show_example_errors():
    plot_example_errors(data, session, correct_pred, prediction , feed_dict_test)

def show_confusion_matrix():
    print_confusion_matrix(data, session, prediction, feed_dict_test, n_classes)