import numpy as np
from utils import mnist_reader
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, InputLayer, add
from sklearn.metrics import accuracy_score


## load the fashion mnist data

x_train, y_train_class = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test_class = mnist_reader.load_mnist('data/fashion', kind='t10k')

num_classes = 10
y_train = to_categorical(y_train_class, num_classes)
y_test = to_categorical(y_test_class, num_classes)


## reshape and normalize the input data

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


#############################################################
# ## CNN Two Layer Deep

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', bias_initializer='RandomNormal', kernel_initializer='random_uniform', 
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

filepath="weights\\cnn_2d.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback = [checkpoint]

model.fit(x_train, y_train, batch_size=256, epochs=50, verbose=1, validation_data=(x_test, y_test), callbacks=callback)

model.load_weights("weights\\cnn_2d.hdf5")
y_predicted_class = model.predict_classes(x_test)

print ('\nTwo Layer Deep CNN Accuracy : ' + str(100 * accuracy_score(y_predicted_class, y_test_class)) + ' %\n')




#############################################################
# ## CNN Two Layer Deep + BatchNorm

model = Sequential()
model.add(InputLayer(input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', bias_initializer='RandomNormal', kernel_initializer='random_uniform', 
                padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

filepath="weights\\cnn_batchnorm_2d.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback = [checkpoint]

model.fit(x_train, y_train, batch_size=256, epochs=50, verbose=1, validation_data=(x_test, y_test), callbacks=callback)

model.load_weights("weights\\cnn_batchnorm_2d.hdf5")
y_predicted_class = model.predict_classes(x_test)

print ('\nTwo Layer Deep Batchnormalized CNN Accuracy : ' + str(100 * accuracy_score(y_predicted_class, y_test_class)) + ' %\n')




#############################################################
# ## CNN Two Layer Deep + BatchNorm + SkipResidualConnections

input_image = Input(shape=(28, 28, 1))

## first layer

x1_batch = BatchNormalization()(input_image)
x1_conv = Conv2D(32, (3, 3), activation='relu', bias_initializer='RandomNormal', kernel_initializer='random_uniform', 
           padding = 'same')(x1_batch)

x1_add = add([x1_batch, x1_conv])
x1_pool = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x1_add)


## second layer

x2_batch = BatchNormalization()(x1_pool)
x2_conv = Conv2D(32, (3, 3), activation='relu', bias_initializer='RandomNormal', kernel_initializer='random_uniform', 
           padding = 'same')(x2_batch)

x2_add = add([x2_batch, x2_conv])
x2_pool = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x2_add)


## ouput layer

x3_batch = BatchNormalization()(x2_pool)

x = Flatten()(x3_batch)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(num_classes, activation='softmax')(x)

model = Model(input_image, preds) 

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

filepath="weights\\cnn_skip_batchnorm_2d.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback = [checkpoint]

model.fit(x_train, y_train, batch_size=256, epochs=50, verbose=1, validation_data=(x_test, y_test), callbacks=callback)

model.load_weights("weights\\cnn_skip_batchnorm_2d.hdf5")
y_preds = model.predict(x_test)

y_predicted_class = np.argmax(y_preds, axis=1)

print ('\nTwo Layer Deep Batchnormalized Skip CNN Accuracy : ' + str(100 * accuracy_score(y_predicted_class, y_test_class)) + ' %\n')