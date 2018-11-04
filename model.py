#!/home/nichlaslr/anaconda3/bin/python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Activation, Conv2D, ELU, ReLU, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D

IMG_SHAPE = (66, 220, 3)


def speed_model():
    inputs = Input(shape=IMG_SHAPE)
    
    timedist = TimeDistributed(Conv2D(36, (3,3), padding="same")(inputs)
    act1 = Activation(ReLU())(conv1)
    conv2 = Conv2D(48, (5,5), strides=(2,2), padding="same")(act1)
    act2 = Activation(ReLU())(conv2)
    drop1 = Dropout()(act2)
    conv3 = Conv2D(64, (5,5), strides=(2,2), padding="same")(drop1)


