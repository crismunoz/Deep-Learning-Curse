#%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

img_size = 28 # Dimensão das imagens de MNIST
img_shape = (img_size, img_size)
 
def set_data_cls(labels):
    return np.array([label.argmax() for label in labels])
    
def plot_images(images, labels, pred=None):
    
    # Pega as 9 primeiras imagens
    images = images[0:9]
    
    # Pega os 9 primeiros rotulos
    cls_true = set_data_cls(labels)[0:9]
    
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            cls_pred = set_data_cls(pred)[0:9]
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

def print_confusion_matrix(data, session, y_pred, feed_dict_test,num_classes):
    # Get the true classifications for the test-set.
    cls_true = set_data_cls(data.test.labels)
    
    # Get the predicted classifications for the test-set.
    pred = session.run(y_pred, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=set_data_cls(pred))

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
def plot_example_errors(data, session, correct_prediction, y_pred, feed_dict_test):
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, pred = session.run([correct_prediction, y_pred],
                                    feed_dict=feed_dict_test)

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    inc_images = data.test.images[incorrect]
    inc_labels = data.test.labels[incorrect]
    # Get the predicted classes for those images.
    inc_pred = pred[incorrect]

    # Get the true classes for those images.
    #cls_true = set_data_test_cls(data.test.labels)[incorrect]
    # Plot the first 9 images.
    plot_images(images=inc_images[0:9],
                labels=inc_labels[0:9],
                pred=inc_pred[0:9])