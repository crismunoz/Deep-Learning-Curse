from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn

LOG=r'train'
tf.logging.set_verbosity(tf.logging.INFO)

############################################ MNIST DATASET #############################################

DATA_TRAIN_SIZE=55000 #Total data:55000
DATA_EVAL_SIZE=2000   #Total data:10000

def get_data_mnist():
    mnist        = learn.datasets.load_dataset("mnist")
    train_data   = mnist.train.images[:DATA_TRAIN_SIZE]  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)[:DATA_TRAIN_SIZE]
    eval_data    = mnist.test.images[:DATA_EVAL_SIZE]  # Returns np.array
    eval_labels  = np.asarray(mnist.test.labels, dtype=np.int32)[:DATA_EVAL_SIZE]    
    return train_data , train_labels , eval_data , eval_labels

######################################## DEFINIÇÃO DO MODELO ###########################################

def model(features, labels, mode):
    """Model CNN."""

    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    
    #----------------------------------------- conv1 ------------------------------------------
    
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=FIXME,
      kernel_size=[FIXME, FIXME],
      padding="same",
      activation=tf.nn.relu)

    #---------------------------------------- pool1 -------------------------------------------
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[FIXME, FIXME], strides=FIXME)
    
    #---------------------------------------- conv2 -------------------------------------------
    
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=FIXME,
      kernel_size=[FIXME, FIXME],
      padding="same",
      activation=tf.nn.relu)
    
    #---------------------------------------- pool2 -------------------------------------------

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[FIXME, FIXME], strides=FIXME)
    
    #--------------------------------------- local3 -------------------------------------------
    
    shape = pool2.get_shape().as_list()
    
    pool2_flat = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])
    
    local3 = tf.layers.dense(inputs=pool2_flat, units=FIXME, activation=tf.nn.relu)
    
    #-------------------------------------- local 4 -------------------------------------------
    
    local4 = tf.layers.dense(inputs=local3, units=FIXME, activation=tf.nn.relu)
    
    #--------------------------------------- softmax -------------------------------------------
    
    logits = tf.layers.dense(inputs=local4, units=10)
    
    prob = tf.nn.softmax(logits, name="softmax_tensor")
    
    cls = tf.argmax(input=logits, axis=1)
    
    return logits , prob , cls

####################################### CONFIGURAÇÃO DO MODELO #########################################

def config_model(features, labels, mode):
    
    logits , prob , cls = model(features, labels, mode)    
    
    loss = None
    train_op = None

    #------- Loss function (para treinamento e teste) -------------------
    
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    #------- Configuração treinamento (TRAIN mode) -----------------------
    
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.01,
        optimizer=FIXME)

    #------ Configuração Predições ----------------------------------------------
    
    predictions = {"classes": cls, "probabilities": prob}

    # Retorna o objeto ModelFnOps
    return model_fn.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

########################################## EXECUTAR MODELO #################################################

def main(unused_argv):
    
    train_data , train_labels , eval_data , eval_labels = get_data_mnist()
    
    # Criação do modelo
    config = tf.contrib.learn.RunConfig(gpu_memory_fraction=0.2)
    model = learn.Estimator(model_fn=config_model, model_dir=LOG, config=config)
    
    # Treinamento
    model.fit(x=train_data, y=train_labels, batch_size=FIXME, steps=FIXME)

    # Configuração das metricas de acuracia pra avaliação
    metrics = {
        "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes")
    }
    
    # Avaliação do modelo e resultados
    eval_results = model.evaluate(x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)
    
    
if __name__ == "__main__":
    tf.app.run(main=main)