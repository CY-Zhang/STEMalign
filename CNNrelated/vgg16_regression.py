import scipy.io as sio
import numpy as np
import glob
import os
import sys
from keras import applications, optimizers, callbacks
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    # load Ronchigrams with noise and emittance values
    x_train_list = []
    y_train_list = []
    input_path = '../../../../TrainingData_CoarseCNN/'
    train_data = np.load(input_path + 'FullRandom_NoAperture_HighCs_60limit_multiNoise_30pA_128pxRonch_x40000.npy')
    train_label = np.load(input_path + 'FullRandom_NoAperture_HighCs_60limit_multiNoise_30pA_emit_x40000.npy')
    train_label = (train_label - np.amin(train_label))/(np.amax(train_label) - np.amin(train_label))

    # pad extra layers to get 3-channel image
    for i in range(40000):
        # if train_label[i] > 0.6:
        #     continue
        frame = scale_range(train_data[i,:,:].astype('float'), 0, 1)
        new_channel = np.zeros(frame.shape)
        img_stack = np.dstack((frame, new_channel, new_channel))
        x_train_list.append(img_stack)
        y_train_list.append(train_label[i])

    x_train = np.concatenate([arr[np.newaxis] for arr in x_train_list])
    y_train = np.asarray(y_train_list,dtype=np.float32)
    del x_train_list, y_train_list, train_data, train_label

    train_top_model(x_train, y_train, 20, 100, 0)

    fully_training(x_train, y_train, 20, 50)

def train_top_model(x_train, y_train, batch_size, epochs, max_index):
    # Train top fully connected layers

    # Parameter setup
    nb_train_samples = x_train.shape[0]
    lr = 1e-4
    decay = 1e-7
    momentum = 0.9
    optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    loss = 'mse'

    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    print('before featurewise center')
        
    datagen = ImageDataGenerator(
        featurewise_center=True,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.05,
        shear_range=0.03)

    datagen.fit(x_train)
    print('made it past featurewise center')
    generator = datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=False)
    print('made it past generator')

    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)

    print('made it past the bottleneck features')
    # np.save('bottleneck_features_train.npy',bottleneck_features_train)

    train_data = bottleneck_features_train
    train_labels = y_train
    print(train_data.shape, train_labels.shape)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation=None))

    model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

    bottleneck_log = 'training_' + str(max_index) + '_bnfeature_log.csv'
    csv_logger_bnfeature = callbacks.CSVLogger(bottleneck_log)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=30, verbose=1, mode='auto')

    model.fit(train_data,train_labels,epochs=epochs,batch_size=batch_size,shuffle=True,
            callbacks=[csv_logger_bnfeature, earlystop],verbose=2,validation_split=0.2)

    with open(bottleneck_log, 'a') as log:
        log.write('\n')
        log.write('input images: ' + '\n')
        log.write('batch_size:' + str(batch_size) + '\n')
        log.write('learning rate: ' + str(lr) + '\n')
        log.write('learning rate decay: ' + str(decay) + '\n')
        log.write('momentum: ' + str(momentum) + '\n')
        log.write('loss: ' + loss + '\n')

    model.save_weights('top_layers.h5')

def fully_training(x_train, y_train, batch_size, epochs):
    # Fine tune the whole model
    # Parameters setup


    lr = 5.5e-5
    decay = 1e-7
    momentum = 0.9
    optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    loss = 'mse'

    # build the model with VGG16 base and trained top connected layers
    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    print('Model loaded')
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.2))
    top_model.add(Dense(1,activation=None))
    top_model.load_weights('top_layers.h5')

    new_model = Sequential()
    for l in model.layers:
        new_model.add(l)
    new_model.add(top_model)

    # Compile the model
    new_model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

    bottleneck_log = 'test.csv'
    csv_logger = callbacks.CSVLogger(bottleneck_log)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.05,
        shear_range=0.03)

    datagen.fit(x_train)

    generator = datagen.flow(
            x_train,
            y_train,
            batch_size=batch_size,
            shuffle=True)

    validation_generator = datagen.flow(
            x_train,
            y_train,
            batch_size=batch_size,
            shuffle=True)

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='auto')

    # Start the training process
    new_model.fit_generator(generator,epochs=epochs,steps_per_epoch=len(x_train) // batch_size,validation_data=validation_generator,validation_steps=(len(x_train)//5)//batch_size,
            callbacks=[csv_logger, earlystop],verbose=2)

    # Save the whole model
    new_model.save_weights('Final_model.h5')

    # make a plot using 1000 data points
    x_validation = x_train[0:1000,:,:,:]
    y_pred = new_model.predict(x_validation, batch_size = 25)
    y_truth = y_train[0:1000]
    y_min = np.amin(y_truth)
    y_max = np.amax(y_truth)

    fig = plt.figure()
    plt.scatter(y_truth, y_pred)
    plt.xlabel('Truth', fontsize = 16)
    plt.ylabel('Prediction', fontsize = 16)
    plt.xlim([y_min,y_max])
    plt.ylim([y_min,y_max])
    plt.plot(np.linspace(y_min,y_max,num=1000),np.linspace(y_min,y_max,num=1000))
    plt.savefig('Validation.png')

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input    

# step 4 make predictions using experiment results

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    gpus = tf.config.experimental.list_physical_devices('GPU')
    os.environ["CUDA_VISIBLE_DEVICES"]="0" # specify which GPU to use
    if gpus:
      try:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      except RuntimeError as e:
        print(e)
    main()
