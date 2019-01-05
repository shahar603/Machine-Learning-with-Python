import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

# TFLearn stuff
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


import matplotlib.pyplot as plt


#import tensorflow as tf
#tf.reset_default_graph()




TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMAGE_SIZE = 51
LR = 1e-3

MODEL_NAME = 'dogsvscats--{}--{}--{}.model'.format(LR, '2conv-basic-5', IMAGE_SIZE)



def label_image(img: str):
    word_label = img.split('.')[0]

    if word_label == 'cat':
        return [1, 0]
    else:
        return [0, 1]





def create_train_data():
    training_data = []

    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_image(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))

        training_data.append([np.array(img), np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data




def process_test_data():
    testing_data = []

    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
        testing_data.append([np.array(img), img_num])

    np.save('testing_data.npy', testing_data)
    return testing_data



def create_model():
    convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet,
                         optimizer='adam',
                         learning_rate=LR,
                         loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    return model




def predict(model, path):
    image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
    data = image.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        return 'dog'

    return 'cat'




#train_data = create_train_data()
train_data = np.load('train_data.npy')

model = create_model()

if os.path.exists(MODEL_NAME + '.meta'):
    model.load(MODEL_NAME)
    print('loaded model')


train = train_data[:-500]
test = train[-500:]




X = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
test_y = np.array([i[1] for i in test])


model.fit({'input': X},
          {'targets': y},
          n_epoch=15,
          validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500,
          show_metric=True,
          run_id=MODEL_NAME)

model.save(MODEL_NAME)



#test_data = process_test_data()
test_data = np.load('testing_data.npy')


fig = plt.figure()

for num, data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    # model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()