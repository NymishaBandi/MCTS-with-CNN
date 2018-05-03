import copy as cp
import random
import numpy as np
import turtle
import time
import os,sys
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image, ImageFilter
import io

# import cv2

random.seed(1234)
np.random.seed(1234)

# number = input("Enter the number you want to draw")
number = 0
print(number)


class State():
    def next_state(self,action):
        raise NotImplementedError()

    def reward(self,action,next_state):
        raise NotImplementedError()


class Canvas(State):

    def __init__(self,obs=turtle.getscreen()):
        self.state = obs
        self.nspeed = 1
        self.smin = 0
        self.smax = 1
        self.nrot = 8
        self.rmax = 45
        self.rinc = self.rmax*2/self.nrot
        self.n_act = self.nspeed * self.nrot
        self.actions ={}
        speed = range(self.smin,self.smax)
        rot = np.arange(-self.rmax,self.rmax,self.rinc)
        for i in speed:
            for j in rot:
                self.actions.update({(i,j):{'nov':0,'Q':0}})
        self.term = False
        turtle.penup()
        turtle.ht()
        

    def get_actions(self,state):
        speed = range(self.smin,self.smax)
        rot = np.arange(-self.rmax,self.rmax,self.rinc)
        for i in speed:
            for j in rot:
                self.actions.update({(i,j):{'nov':0,'Q':0}})
        return self.actions

    def next_state(self,action):
        turtle.tracer(100)
        turtle.speed(action[0])
        turtle.pendown()
        turtle.right(action[1])
        turtle.forward(10)
        return Canvas(turtle.getscreen())
        #draw on the canvas everytime this is called

        
    def reward(self,model):
        #generate an image and pass to CNN to get reward
        # turtle.update()
        # os.system("rm image.bmp")
        turtle.update()
        mod=model
        ts = turtle.getscreen()
        ts.getcanvas().postscript(file="image.eps")
        reward = predict(mod,"image.eps")
        return reward

    def env_reset(self,path):
        turtle.reset()
        # draw the current best actions
        length = len(path)
        if length>0:
            print('Updating')
            turtle.penup()
            turtle.goto(-130,150)
            turtle.pendown()
            for i in range(length):
                action = path[i]
                turtle.speed(action[0])
                turtle.pendown()
                turtle.right(action[1])
                turtle.forward(10)
        else:
            turtle.penup()
            turtle.goto(-130,150)
            turtle.pendown()
        return 1

def model():
    batch_size = 128
    num_classes = 10
    epochs = 8

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    return model


def predict(mod,img):
    # size = 256, 256
    # im = Image.open(img).convert('LA')
    # im.thumbnail(size, Image.ANTIALIAS)
    # im.save('greyscale.png')
    image=imageprepare('image.eps')

    # cv2.imshow('image',image)
    # print (image.shape)
    # image = cv2.resize(image, (28, 28))
    model=mod
    prediction = np.argmax(model.predict(image))
    if prediction == number:
        return 1
    else:
        return 0

def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    newImage.save("sample.png")

    tv = np.array(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    tva = np.array(tva).reshape(1,28,28,1)
    # print(tva.shape)
    return tva