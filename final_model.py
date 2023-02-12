import tensorflow as tf
import pandas as pd
import numpy as np

# Create generator to get data
dataGenerator = keras.preprocessing.image.ImageDataGenerator()

test = dataGenerator.flow_from_directory("data/real_vs_fake/real-vs-fake/test", 
                                         class_mode = 'binary', 
                                         batch_size = 32)
train = dataGenerator.flow_from_directory("data/real_vs_fake/real-vs-fake/train", 
                                          class_mode = 'binary', 
                                          batch_size = 32)
valid = dataGenerator.flow_from_directory("data/real_vs_fake/real-vs-fake/valid", 
                                          class_mode = 'binary',
                                          batch_size = 32)


# Create the model:
def ProjectNet_v3():
    m = keras.models.Sequential()
    m.add(keras.layers.Conv2D(filters= 32, input_shape = (256,256,3), kernel_size=(1,1), strides=(2,2), padding = "same"))
    m.add(keras.layers.BatchNormalization())
    
    m.add(keras.layers.Conv2D(filters= 32, kernel_size=(7,7), strides=(2,2), padding = "same"))
    m.add(keras.layers.BatchNormalization())
    m.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    
    m.add(keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
    m.add(keras.layers.BatchNormalization())
    m.add(keras.layers.Activation('relu'))
    
    m.add(keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
    m.add(keras.layers.BatchNormalization())
    m.add(keras.layers.Activation('relu')) 
    
    #Inception
    m.add(keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1,1), padding='same'))
    m.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same'))
    m.add(keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))


    m.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same'))
    m.add(keras.layers.BatchNormalization())
    m.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides = (1,1)))
    
    
    m.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    m.add(keras.layers.BatchNormalization())
    m.add(keras.layers.Activation('relu'))
          
    m.add(keras.layers.GlobalMaxPooling2D())
    m.add(keras.layers.Dense(units= 128))
    m.add(keras.layers.Dropout(0.3))
    m.add(keras.layers.Dense(units= 32))
    m.add(keras.layers.Dense(units=1, kernel_initializer="glorot_uniform"))
    m.add(keras.layers.Activation('sigmoid'))
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                            loss='binary_crossentropy',
                                            metrics=['binary_accuracy'])
    
    
    return(m)

#Compile
model_v3 = ProjectNet_v3()

#Fit
modelHistory = model_v3.fit(train, batch_size=32, epochs=2, validation_data=valid)

#Evalute
model_v3.evaluate(test)

#Save
model_v3.save('model_v3.h5')