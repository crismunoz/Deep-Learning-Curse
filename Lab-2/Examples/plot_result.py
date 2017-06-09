from Helper_functions import plot_images,plot_example_errors,print_confusion_matrix

def show_example_errors():
    plot_example_errors(data, session, correct_pred, pred , feed_dict_test)

def show_confusion_matrix():
    print_confusion_matrix(data, session, pred, feed_dict_test, n_classes)