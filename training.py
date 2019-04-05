#!/usr/bin/env python3
from preprocessor import PreProcessor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import model 


pre = PreProcessor()


def generate_training_data(data, batch_size = 16):
    image_batch = np.zeros((batch_size, 66, 220, 3)) # nvidia input params
    label_batch = np.zeros((batch_size))
    while True:
        for i in range(batch_size):
            idx = np.random.randint(1, len(data) - 1)
            
            row_now = data.iloc[[idx]].reset_index()
            row_prev = data.iloc[[idx - 1]].reset_index()
            row_next = data.iloc[[idx + 1]].reset_index()
            
            time_now = row_now[1].values[0]
            time_prev = row_prev[1].values[0]
            time_next = row_next[1].values[0]
            
            if time_now - time_prev > 0 and 0.0000001 < time_now - time_prev < 0.58: # 0.578111 is highest diff i have seen
                row1 = row_prev
                row2 = row_now
                
            elif time_next - time_now > 0 and 0.0000001 < time_next - time_now < 0.58:
                row1 = row_now
                row2 = row_next
                
            x1, y1 = pre.preprocess_image_from_path(row1[0].values[0],row1[2].values[0]) 
            x2, y2 = pre.preprocess_image_from_path(row2[0].values[0],row2[2].values[0])
           
            img_diff = pre.optical_flow(x1, x2)
            y = np.mean([y1, y2])
            
            image_batch[i] = img_diff
            label_batch[i] = y
            
        yield shuffle(image_batch, label_batch)


def generate_validation_data(data):
    while True:
        for idx in range(1, len(data) - 1): 
            row_now = data.iloc[[idx]].reset_index()
            row_prev = data.iloc[[idx - 1]].reset_index()
            row_next = data.iloc[[idx + 1]].reset_index()
            
            time_now = row_now[1].values[0]
            time_prev = row_prev[1].values[0]
            time_next = row_next[1].values[0]
            
            if time_now - time_prev > 0 and 0.0000001 < time_now - time_prev < 0.58:
                row1 = row_prev
                row2 = row_now
                
            elif time_next - time_now > 0 and 0.000001 < time_next - time_now < 0.58:
                row1 = row_now
                row2 = row_next

            x1, y1 = pre.preprocess_image_valid_from_path(row1[0].values[0], row1[2].values[0])
            x2, y2 = pre.preprocess_image_valid_from_path(row2[0].values[0], row2[2].values[0])
            
            img_diff = pre.optical_flow(x1, x2)
            img_diff = img_diff.reshape(1, img_diff.shape[0], img_diff.shape[1], img_diff.shape[2])
            y = np.mean([y1, y2])
            
            speed = np.array([[y]])
            yield img_diff, speed



if __name__ == "__main__":
    filepath = 'model-weights.h5'
    df = pd.read_csv("./processed.csv", header = None)
    pre = PreProcessor()
    train, test = pre.shuffle_frame_pairs(df)
    size_test = len(test.index)
    size_train = len(train.index)
    #print(size_test)
    #print(size_train)
    dl_model = model.speed_model()
    earlyStopping = EarlyStopping(monitor='val_loss', 
                              patience=2, 
                              verbose=1, 
                              min_delta = 0.23,
                              mode='min',)
    modelCheckpoint = ModelCheckpoint(filepath, 
                                      monitor = 'val_loss', 
                                      save_best_only = True, 
                                      mode = 'min', 
                                      verbose = 1,
                                     save_weights_only = True)
    callbacks_list = [modelCheckpoint, earlyStopping]
    train_generator = generate_training_data(train)
    test_generator = generate_validation_data(test)
    history = dl_model.fit_generator(
            train_generator, 
            steps_per_epoch = 555,
            epochs = 25,
            callbacks = callbacks_list,
            verbose = 1,
            validation_data = test_generator,
            validation_steps = size_test)

    print(history)
