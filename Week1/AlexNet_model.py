from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

############################################ CIFAR10 DATASET #############################################

import cifar10

def get_data(eval_data):
    return lambda: cifar10.inputs(eval_data)

        
######################################## DEFINIÇÃO DO MODELO ###########################################

def AlexNet_Model(features, labels, mode):
    
    input_layer = tf.reshape(features, [-1, 24, 24, 3])

    #----------------------------------------- conv1 ------------------------------------------
    
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=FIXME,
      kernel_size=[FIXME, FIXME],
      padding="same",
      activation=tf.nn.relu)

    #----------------------------------------- pool1 ------------------------------------------

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[FIXME, FIXME], strides=FIXME)

    #----------------------------------------- norm1 ------------------------------------------
    
    norm1 = tf.nn.lrn(pool1, depth_radius=FIXME, bias=FIXME, alpha=FIXME, beta=FIXME)

    #----------------------------------------- conv2 ------------------------------------------
    
    conv2 = tf.layers.conv2d(
      inputs=norm1,
      filters=FIXME,
      kernel_size=[FIXME, FIXME],
      padding="same",
      activation=tf.nn.relu)
    
    #----------------------------------------- pool2 ------------------------------------------
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[FIXME, FIXME], strides=FIXME)
    
    #----------------------------------------- norm2 ------------------------------------------

    norm2 = tf.nn.lrn(pool2, depth_radius=FIXME, bias=FIXME, alpha=FIXME, beta=FIXME)

    #----------------------------------------- conv3 ------------------------------------------
    
    conv3 = tf.layers.conv2d(
      inputs=norm2,
      filters=FIXME,
      kernel_size=[FIXME, FIXME],
      padding="same",
      activation=tf.nn.relu)
    
    #----------------------------------------- conv4 ------------------------------------------
    
    conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=FIXME,
      kernel_size=[FIXME, FIXME],
      padding="same",
      activation=tf.nn.relu)
    
    #----------------------------------------- conv5 ------------------------------------------
    
    conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=FIXME,
      kernel_size=[FIXME, FIXME],
      padding="same",
      activation=tf.nn.relu)
    
    #----------------------------------------- pool5 ------------------------------------------
    
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[FIXME, FIXME], strides=FIXME)
    
    #------------------------------------ flatten ---------------------------------------------
    
    shape = pool5.get_shape().as_list()

    pool5_flat = tf.reshape(pool5, [-1, shape[1] * shape[2] * shape[3]])

    #----------------------------------------- local6 ------------------------------------------
    
    local6 = tf.layers.dense(inputs=pool5_flat, units=FIXME, activation=tf.nn.relu)

    #----------------------------------------- local7 -------------------------------------------
    
    local7 = tf.layers.dense(inputs=local6, units=FIXME, activation=tf.nn.relu)

    #---------------------------------------- softmax -------------------------------------------
    
    logits = tf.layers.dense(inputs=local7, units=10)

    prob = tf.nn.softmax(logits, name="softmax_tensor")
    
    cls = tf.argmax(input=logits, axis=1)
    
    return logits , prob , cls

####################################### CONFIGURAÇÃO DO MODELO #########################################

def AlexNet_Model_fn(features, labels, mode):
    
    logits , prob , cls = AlexNet_Model(features, labels, mode)    

    loss = None
    train_op = None

    #------- Loss function (para treinamento e teste) -------------------
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int64), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    #------- Configuração treinamento (TRAIN mode) -----------------------
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001))

    #------ Configuração Predições ----------------------------------------------
    predictions = {"classes": cls, "probabilities": prob}

    # Retorna o objeto ModelFnOps
    return model_fn.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

########################################## EXECUTAR MODELO #################################################

def main(unused_argv):
        
    # Criação do modelo
    model = learn.Estimator(model_fn=AlexNet_Model_fn, model_dir="/tmp/mnist_convnet_model")

    # Treinamento
    model.fit(input_fn=get_data(eval_data=False), steps=FIXME)

    # Configuração das metricas de acuracia pra avaliação
    metrics = {
        "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes")
    }

    # Avaliação do modelo e resultados
    eval_results = model.evaluate(input_fn=get_data(eval_data=True), metrics=metrics, steps=50)
    print(eval_results)
    
    
if __name__ == "__main__":
    tf.app.run(main=main)