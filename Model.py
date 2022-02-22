from tensorflow.keras.layers import Input, Conv2D, concatenate, Activation, Dropout, BatchNormalization, Subtract, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
import tensorflow as tf


class CNNSpeckleFilter:
    def __init__(self, input_shape, n_layers):
        self.model = self.__build_model(input_shape, n_layers)

        def custom_loss(y_true, y_pred):
            ssim = 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
            mse = K.mean(K.square(y_pred - y_true), axis=-1)
            tv = tf.reduce_mean(tf.image.total_variation(y_pred))
            
            return ssim + mse + 0.00001*tv

        self.optimizer = Adam(learning_rate=0.002)#, 
                              #beta_1=0.9, 
                              #beta_2=0.999, 
                              #epsilon=None, 
                              #decay=0.0, 
                              #amsgrad=False)

        self.model.compile(
            loss =  custom_loss, #'binary_crossentropy',#custom_loss, 
            optimizer = self.optimizer, 
            metrics = ['mse', 'mae'])

    def __build_model(self, input_shape, n_layers):
        n_filter = 64
        kernel_size = (3,3)
        stride = (1,1)

        # Input layer
        x_input = Input(shape = input_shape)
        x = Conv2D(filters = n_filter, kernel_size = kernel_size, strides = stride, padding = 'same', use_bias=True)(x_input)
        x = Activation('relu')(x)
        # Repeated layers
        
        for i in range(n_layers):
            x = Conv2D(filters = n_filter, kernel_size = kernel_size, strides = stride, padding = 'same', use_bias=True)(x)
            #x = MaxPooling2D((2,2), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        # Conversion layer
        x = Conv2D(filters = 1, kernel_size = kernel_size, strides = stride, padding = 'same', use_bias=True)(x)
        x = Activation('tanh')(x)
    
        skip = Subtract()([x_input,x])

        #x = Conv2D(filters = 1, kernel_size = kernel_size, strides = stride, padding = 'same', use_bias=True)(skip)
        x_output = Activation('sigmoid')(skip)
        
        model = Model(inputs = x_input, outputs = x_output, name = 'CNNSpeckleFilter')
        return model

    def train_model(self, epochs, train_gen, val_gen, train_step, val_step):
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        def lr_scheduler(epoch, lr):
            decay_rate = 0.6
            decay_step = 3
            if epoch % decay_step == 0 and epoch:
                return lr * decay_rate
            return lr

        callbacks = [
          LearningRateScheduler(lr_scheduler, verbose=3),
          es
        ]

        history = self.model.fit(
            train_gen,
            steps_per_epoch = train_step,
            validation_data=val_gen,
            validation_steps=val_step,
            epochs = epochs,
            callbacks=callbacks
        )

        return history
