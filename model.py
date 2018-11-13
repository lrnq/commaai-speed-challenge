#!/home/nichlaslr/anaconda3/bin/python
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, ELU, TimeDistributed, Flatten, Dropout, Lambda
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import MaxPooling2D

IMG_SHAPE = (66, 220, 3)


def speed_model():
    inputs = Input(shape=IMG_SHAPE)
    inputs1 = Lambda(lambda x: x/ 127.5 - 1, input_shape = IMG_SHAPE)(inputs)

    conv1 = Conv2D(24, (5, 5), padding="valid")(inputs1)
    act1 = Activation(ELU())(conv1)
    conv2 = Conv2D(36, (5, 5), strides=(2, 2), padding="valid")(act1)
    act2 = Activation(ELU())(conv2)
    drop1 = Dropout(0.5)(act2)
    conv3 = Conv2D(48, (5, 5), strides=(2, 2), padding="valid")(drop1)
    act3 = Activation(ELU())(conv3)
    conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding="valid")(act3)
    act4 = Activation(ELU())(conv4)
    conv5 = Conv2D(64, (3, 3), padding="valid")(act4)


    flat1 = Flatten()(conv5)
    act4 = Activation(ELU())(flat1)
    dense1 = Dense(100)(act4)
    act5 = Activation(ELU())(dense1)
    dense2 = Dense(50)(act5)
    act6 = Activation(ELU())(dense2)
    dense4 = Dense(10)(act6)
    act8 = Activation(ELU())(dense4)
    output = Dense(1)(act8)

    model = Model(inputs, output)
    adam = Nadam()
    model.compile(optimizer=adam, loss='mse')

    print(model.summary())
    return model



def nvidia_model():

    model = Sequential()
    # normalization    
    # perform custom normalization before lambda layer in network
    model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = IMG_SHAPE))

    model.add(Conv2D(24, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv1'))
    
    
    model.add(ELU())    
    model.add(Conv2D(36, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv2'))
    
    model.add(ELU())    
    model.add(Conv2D(48, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv3'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), 
                            strides = (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv4'))
    
    model.add(ELU())              
    model.add(Conv2D(64, (3, 3), 
                            strides= (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv5'))
              
              
    model.add(Flatten(name = 'flatten'))
    model.add(ELU())
    model.add(Dense(100, kernel_initializer = 'he_normal', name = 'fc1'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer = 'he_normal', name = 'fc2'))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer = 'he_normal', name = 'fc3'))
    model.add(ELU())
    
    # do not put activation at the end because we want to exact output, not a class identifier
    model.add(Dense(1, name = 'output', kernel_initializer = 'he_normal'))
    
    adam = Nadam()
    model.compile(optimizer = adam, loss = 'mse')

    return model


if __name__ == "__main__":
    print("These are models from Nvidia end to end paper")
