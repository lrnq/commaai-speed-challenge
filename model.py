#!/home/nichlaslr/anaconda3/bin/python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Activation, Conv2D, ReLU, TimeDistributed, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D

IMG_SHAPE = (66, 220, 3)


def speed_model():
    inputs = Input(shape=IMG_SHAPE)
    
    conv1 = Conv2D(36, (3,3), padding="same")(inputs)
    act1 = Activation(ReLU())(conv1)
    conv2 = Conv2D(48, (5,5), strides=(2,2), padding="same")(act1)
    act2 = Activation(ReLU())(conv2)
    drop1 = Dropout(0.4)(act2)
    conv3 = Conv2D(64, (5,5), strides=(2,2), padding="same")(drop1)
    act3 = Activation(ReLU())(conv3)
    conv4 = Conv2D(128, (3,3), strides=(1,1), padding="same")(act3)

   
    flat1 = Flatten()(conv4)
    act4 = Activation(ReLU())(flat1)
    dense1 = Dense(100)(act4)
    act5 = Activation(ReLU())(dense1)
    dense2 = Dense(50)(act5)
    act6 = Activation(ReLU())(dense2)
    dense4 = Dense(10)(act6)
    act8 = Activation(ReLU())(dense4)
    output = Dense(1)(act8)

    model = Model(inputs, output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = adam, loss = 'mse')

    print(model.summary())
    return model


if __name__ == "__main__":
    speed_model()
