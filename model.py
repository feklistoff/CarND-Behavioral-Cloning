import cv2
import random
import pandas as pd
import matplotlib.image as mpimg
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Lambda, Dense, Flatten, Dropout, Convolution2D, ELU
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint


# constants
SIZE = 64
BATCH_SIZE = 64


# read csv logs
columns = ['center', 'left', 'right', 'angle', 'throttle', 'brake', 'speed']
rows = pd.read_csv('./data/driving_log.csv', names=columns)
angles = rows['angle'].tolist()
# NOTE: path format to images: /home/usr_name/CarND-Behavioral-Cloning-P3/data/IMG/some_img.jpg
center = rows['center'].tolist()
left = rows['left'].tolist()
right = rows['right'].tolist()


# add steering angle correction/recovery
recovery = 0.27
turn_correction = 0.2
images_data = []
angles_data = []
for i in range(len(angles)):
    images_data.extend([center[i], left[i], right[i]])
    if angles[i] >= 0.35 and angles[i] < 0.55: # turn right
        r = recovery * (1 + random.uniform(-0.1, 0.1)) # adding some noise
        t = turn_correction * (1 + random.uniform(-0.1, 0.1))
        angles_data.extend([angles[i], angles[i] + r, angles[i] + t])
    if angles[i] <= -0.35 and angles[i] > -0.55: # turn left
        r = recovery * (1 + random.uniform(-0.1, 0.1))
        t = turn_correction * (1 + random.uniform(-0.1, 0.1))
        angles_data.extend([angles[i], angles[i] - t, angles[i] - r])
    if angles[i] > -0.35 and angles[i] < 0.35: # straight
        r = recovery * (1 + random.uniform(-0.1, 0.1))
        angles_data.extend([angles[i], angles[i] + r, angles[i] - r])
    if angles[i] >= 0.55: # sharp turn right
        r = 0.35
        angles_data.extend([angles[i], angles[i] + r, angles[i] + r])
    if angles[i] <= -0.55: # sharp turn left
        r = 0.35
        angles_data.extend([angles[i], angles[i] - r, angles[i] - r])


# split data
shuffle(images_data, angles_data)
imgs_train, imgs_valid, angles_train, angles_valid = train_test_split(images_data,
                                                                      angles_data, test_size=0.3)


# data preprocessing pipeline
def crop_resize(img):
    # check that size always the same
    if img.shape[0] > 160 or img.shape[0] < 160:
        img = cv2.resize(img, (160, 320))
    return cv2.resize(img[60:140], (SIZE, SIZE))

def correct_contrast(img):
    for i in range(img.shape[2]):
        img[:, :, i] = cv2.equalizeHist(img[:, :, i])
    return img

def random_light(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    light = random.uniform(0.1, 0.9)
    img[:, :, 2] = light * img[:, :, 2]
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

def random_shift(img):
    if random.randint(0, 1) == 0:
        return img
    h, w, c = img.shape
    shape = (w, h)
    M = np.float32([[1, 0, random.randint(-SIZE // 7, SIZE // 7)],
                    [0, 1, random.randint(-SIZE // 9, SIZE // 9)]])
    return cv2.warpAffine(img, M, shape)

def random_rotation(img):
    if random.randint(0, 1) == 0:
        return img
    angle = random.uniform(-20, 20)
    height, width, ch = img.shape
    shape = (width, height)
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(img, M, shape)

def random_shadow(img):
    if random.randint(0, 1) == 0:
        return img
    # set top x
    x_top = random.randint(0, img.shape[1])
    # set bottom x
    x_bot = random.randint(0, img.shape[1])
    if x_top >= img.shape[1] // 2:
        x_bot = random.randint(0, img.shape[1] // 2)
    if x_top > img.shape[1] // 2:
        x_bot = random.randint(img.shape[1] // 2, img.shape[1])
    # set corner x
    x3 = x4 = random.choice([0, img.shape[1]])
    # set y1, y2
    y1 = 0
    y2 = img.shape[0]
    # draw shadow
    overlay = np.copy(img)
    pts = np.array([[x_top, y1], [x3, y1], [x4, y2], [x_bot, y2]], np.int32)
    shadow = cv2.fillPoly(overlay, [pts], (0, 0, 0))
    alfa = random.uniform(0.2, 0.8)
    return cv2.addWeighted(shadow, alfa, img, 1 - alfa, 0)

def flip(img, angle):
    if random.randint(0, 1) == 0:
        return img, angle
    img = np.fliplr(img)
    angle = -angle
    return img, angle

def to_yuv(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


# generators
def train_generator(batch_size):
    while True:  # loop forever so the generator never terminates
        shuffle(imgs_train, angles_train)
        batch_imgs = []
        batch_angles = []
        for i in range(batch_size):
            random_index = random.randint(0,len(imgs_train)-1)
            angle = angles_train[random_index] * (1 + random.uniform(-0.1, 0.1)) # add noise
            img = mpimg.imread(imgs_train[random_index])
            img = crop_resize(img)
            img = correct_contrast(img)
            img = random_light(img)
            img = random_shadow(img)
            img = random_shift(img)
            img = random_rotation(img)
            img = to_yuv(img)
            img, angle = flip(img, angle)
            batch_imgs.append(img)
            batch_angles.append(angle)
        yield (np.array(batch_imgs), np.array(batch_angles))


def valid_generator(batch_size):
    while True:  # loop forever so the generator never terminates
        shuffle(imgs_valid, angles_valid)
        batch_imgs = []
        batch_angles = []
        for i in range(batch_size):
            random_index = random.randint(0,len(imgs_valid)-1)
            angle = angles_train[random_index]
            img = mpimg.imread(imgs_valid[random_index])
            img = crop_resize(img)
            img = correct_contrast(img)
            img = to_yuv(img)
            batch_imgs.append(img)
            batch_angles.append(angle)
        yield (np.array(batch_imgs), np.array(batch_angles))


# generate
generator_train = train_generator(BATCH_SIZE)
generator_valid = valid_generator(BATCH_SIZE)


# parameters
samples_per_epoch = (len(imgs_train) // BATCH_SIZE) * BATCH_SIZE
shape = (SIZE, SIZE, 3)


# network architecture
def nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 250) - 0.5, input_shape=shape))

    model.add(Convolution2D(24, 5, 5, init='he_normal', subsample=(2, 2),
                            border_mode='valid', W_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Convolution2D(36, 5, 5, init='he_normal', subsample=(2, 2),
                            border_mode='valid', W_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Convolution2D(48, 5, 5, init='he_normal', subsample=(2, 2),
                            border_mode='valid', W_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1),
                            border_mode='valid', W_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1),
                            border_mode='valid', W_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Flatten())

    model.add(Dense(100, init='he_normal'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(0.7))

    model.add(Dense(50, init='he_normal'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(0.7))

    model.add(Dense(10, init='he_normal'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(0.4))

    model.add(Dense(1))
    return model

model = nvidia()

# train and save
adam = Adam(lr=0.001)

# helper function for step decay
import keras.backend as K
def scheduler(epoch):
    if epoch == 0:
        print("learning rate:", K.get_value(model.optimizer.lr))
    if (epoch+1) % 5 == 0:
        rate = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, rate * 0.3)
        print("learning rate:", K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)

learning_rate_decay = LearningRateScheduler(scheduler)
early_stop = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

model.compile(loss='mse', optimizer=adam)
model.summary()

history = model.fit_generator(generator_train,
                    samples_per_epoch=samples_per_epoch*2,
                    validation_data=generator_valid,
                    nb_val_samples=len(imgs_valid),
                    nb_epoch=20,
                    callbacks=[checkpoint, early_stop, learning_rate_decay])

model.save('model.h5')
