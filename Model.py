from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, Subtract, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler


class CNNSpeckleFilter:
    def __init__(self, input_shape, n_layers):
        self.model = self.__build_model(input_shape, n_layers)

        self.optimizer = Adam(lr=0.0002)
        self.model.compile(
            loss = 'mse', 
            optimizer = self.optimizer, 
            metrics = ['mae'])

    def __build_model(self, input_shape, n_layers):
        n_filter = 64
        kernel_size = (3,3)
        stride = (1,1)

        # Input layer
        x_input = Input(shape = input_shape)
        x = Conv2D(filters = n_filter, kernel_size = kernel_size, strides = stride, padding = 'same')(x_input)
        x = Activation('relu')(x)
        # Repeated layers
        
        for i in range(n_layers):
            x = Conv2D(filters = n_filter, kernel_size = kernel_size, strides = stride, padding = 'same')(x)
            #x = MaxPooling2D((2,2), padding='same')(x)
            #x = BatchNormalization()(x)
            x = Activation('relu')(x)

        #for i in range((n_layers//2)):
        #    x = Conv2D(filters = n_filter, kernel_size = kernel_size, strides = stride, padding = 'same')(x)
        #    x = BatchNormalization()(x)
        #    x = Activation('relu')(x)
        #    x = UpSampling2D((2,2))(x)

        # Conversion layer
        x = Conv2D(filters = 1, kernel_size = kernel_size, strides = stride, padding = 'same')(x)
        x = Activation('relu')(x)
    
        skip = Subtract()([x_input,x])

        x_output = Activation('sigmoid')(skip)
        
        model = Model(inputs = x_input, outputs = x_output, name = 'CNNSpeckleFilter')
        
        return model
    
    def train_model(self, epochs, train_gen, val_gen, train_step, val_step):
        #es = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        def lr_scheduler(epoch, lr):
          decay_rate = 0.85
          decay_step = 3
          if epoch % decay_step == 0 and epoch:
              return lr * decay_rate
          return lr

        callbacks = [
          LearningRateScheduler(lr_scheduler, verbose=1)
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