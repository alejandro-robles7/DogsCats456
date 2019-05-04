# %%
## import some packages to use
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import gc  # Gabage collector for cleaning deleted data from memory
import matplotlib.image as mpimg
import cv2
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img


def get_data_paths(base_path, data_path, sample_size):

    train_dir, test_dir = base_path + data_path[0], base_path + data_path[1]
    train_fullpath, test_fullpath = train_dir + '/{}', test_dir + '/{}'

    train_dogs = [train_fullpath.format(i) for i in os.listdir(train_dir) if 'dog' in i]  # get dog images
    train_cats = [train_fullpath.format(i) for i in os.listdir(train_dir) if 'cat' in i]  # get cat images

    test_imgs = [test_fullpath.format(i) for i in os.listdir(test_dir)]  # get test images

    train_imgs = train_dogs[:sample_size] + train_cats[:sample_size]  # slice the dataset and use 2000 in each class
    random.shuffle(train_imgs)  # shuffle it randomly

    # Clear list that are useless
    del train_dogs
    del train_cats
    gc.collect()  # collect garbage to save memory

    return train_imgs, test_imgs

def get_directory():
    dirpath = os.getcwd()
    print("current directory is : " + dirpath)
    foldername = os.path.basename(dirpath)
    print("Directory name is : " + foldername)


def plot_sample(images, sample_size):
    for ima in images[:sample_size]:
        img = mpimg.imread(ima)
        imgplot = plt.imshow(img)
        plt.show()


# A function to read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images, nrows, ncolumns):
    """
    Returns two arrays:
        X is an array of resized images
        y is an array of labels
    """
    X = []  # images
    y = []  # labels

    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns),
                            interpolation=cv2.INTER_CUBIC))  # Read the image
        # get the labels
        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)

    return np.array(X), np.array(y)



def plot_multiple_pics(images, columns=5):
    # Lets view some of the pics
    plt.figure(figsize=(20, 10))
    for i in range(columns):
        plt.subplot(columns / columns + 1, columns, i + 1)
        plt.imshow(images[i])

def plot_labels(labels):
    # Lets plot the label to be sure we just have two class
    sns.countplot(labels)
    plt.title('Labels for Cats and Dogs')


def split_data(x, y, test_size=0.20, random_state=2):
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=random_state)
    print("Shape of train images is:", X_train.shape)
    print("Shape of validation images is:", X_val.shape)
    print("Shape of labels is:", y_train.shape)
    print("Shape of labels is:", y_val.shape)
    return X_train, X_val, y_train, y_val


def init_model(input_shape, dropout=0.5):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))  # Dropout for regularization
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def process_training_images(x_train, y_train, x_val, y_val, x_test, batch_size):
    train_datagen = ImageDataGenerator(rescale=1. / 255,  # Scale the image between 0 and 1
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255)  # We do not augment validation data. we only perform rescale

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)
    test_generator = test_datagen.flow(x_test, batch_size=batch_size)

    return train_generator, val_generator, test_generator

def plot_history(history):
    # lets plot the train and val curve
    # get the details form the history object
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Train and validation accuracy
    plt.plot(epochs, acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.title('Training and Validation accurarcy')
    plt.legend()

    plt.figure()
    # Train and validation loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.show()


def predict_and_plot(model, test_generator, columns=5):
    text_labels = []
    plt.figure(figsize=(30, 20))
    for i, batch in enumerate(test_generator):
        pred = model.predict(batch)
        if pred > 0.5:
            text_labels.append('dog')
        else:
            text_labels.append('cat')
        plt.subplot(5 / columns + 1, columns, i + 1)
        plt.title('This is a ' + text_labels[i])
        imgplot = plt.imshow(batch[0])
        i += 1
        if i % 10 == 0:
            break
    plt.show()
    return text_labels


if __name__ == '__main__':

    # ********* Getting Data Ready *********

    # Paths
    base_path = '../data/perrosygatos/'
    data_path = ['train', 'test1']

    # Lets declare our image dimensions we are using coloured images.
    n_rows = 150
    n_columns = 150
    channels = 3

    # Sample Data for fast prototyping
    sample_size = 2000

    # Getting Data
    train_images, test_images = get_data_paths(base_path, data_path, sample_size=sample_size)
    X, y = read_and_process_image(train_images, n_rows, n_columns)
    X_test, y_test = read_and_process_image(test_images[:sample_size])
    plot_labels(y)
    print("Shape of train images is:", X.shape)
    print("Shape of labels is:", y.shape)

    # Split Data
    X_train, X_val, y_train, y_val = split_data(X, y)
    ntrain = len(X_train)
    nval = len(X_val)


    # clearing memory
    del train_images
    del X
    del y
    gc.collect()


    # ********* Start Modeling *********

    # Parameters
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    batch_size = 32
    dropout = 0.5
    epochs = 64
    learning_rate = 1e-4
    loss_metric = 'binary_crossentropy'
    validation_metric = 'acc'
    input_shape = (n_rows, n_columns, channels)

    model = init_model(input_shape, dropout)
    model.summary()
    model.compile(loss=loss_metric, optimizer=optimizers.RMSprop(lr=learning_rate), metrics=[validation_metric])

    train_generator, validation_generator, test_generator = process_training_images(X_train, y_train,
                                                                                    X_val, y_val,
                                                                                    X_test,
                                                                                    batch_size)

    # The training part
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=ntrain // batch_size,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=nval // batch_size)

    # Save the model
    model.save_weights('model_wieghts.h5')
    model.save('model_keras.h5')

    # Predicted Labels
    predicted_labels = predict_and_plot(model, test_generator)


