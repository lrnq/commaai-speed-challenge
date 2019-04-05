#!/usr/bin/env python3
import os
import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VIDEO_FILE = "./data/train.mp4"
LABELS = "./data/train.txt"
OUTPUT_PATH = "./data_preprocessed/"
FRAME_RATE = 20

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
    print("Directory " , OUTPUT_PATH,  " Created ")

flow_mat = None
image_scale = 0.5
nb_images = 1
win_size = 15
nb_iterations = 2
deg_expansion = 5
STD = 1.3

class PreProcessor:
    def __init__(self):
        pass

    def __str__(self):
        return "Class for the comma.ai speedchallenge 2018"

    def grayscale(self, frame):
        """Method for converting frame to grey scale"""
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    def plot_training_speed(self, data):
        """Method to plot the labels"""
        data = np.loadtxt(data)
        # df = pd.read_csv(data)
        # print(df)
        # X = df[1].values
        # y = df[2].values
        plt.plot(data)
        plt.show()

    def generate_images(self, file_path, label_path, gray=False):
        """Method for saving each frame from the video to disk.
        Input is the file path to the video and the path to the
        file containing the labels (speed). The time for each frame
        is computed and stored in a csv file with speed label and
        image path"""
        # First load in the labels
        labels = np.loadtxt(label_path)

        # Video file
        video_capture = cv2.VideoCapture(file_path)

        # Make sure number of labels is the same as the number of frames
        if (len(labels) == video_capture.get(cv2.CAP_PROP_FRAME_COUNT)):
            print("Labels are equal to number of frames")

        with open('processed.csv', 'w') as file:
            writer = csv.writer(file)
            time_between_frames = 1 / FRAME_RATE
            time_elapsed = 0
            for idx, i in enumerate(labels):
                ret, frame = video_capture.read()
                time_elapsed += time_between_frames
                if gray:
                    frame = self.grayscale(frame)
                if ret:
                    cv2.imshow("frame", frame)
                    image_path = OUTPUT_PATH + str(time_elapsed) + '.jpg'
                    cv2.imwrite(image_path, frame)
                    writer.writerow([image_path, time_elapsed, labels[idx]])
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Stop video
                    break

        file.close()
        video_capture.release()
        cv2.destroyAllWindows()

    def shuffle_frame_pairs(self, dataframe, val_split=1):
        """Method that shuffles pairs of frames from the video.
        Takes a dataframe as an argument, and returns a training
        and validation dataframe. A custom method like this rather
        than a traditional test, train split is used, as we need
        pair of frames, later on to do optical flow"""
        training_data = pd.DataFrame()
        validation_data = pd.DataFrame()
        df_len = dataframe.shape[0]
        for i in range(df_len - 1):
            idx1 = np.random.randint(df_len - 1)
            idx2 = idx1 + 1
            row1 = dataframe.iloc[[idx1]].reset_index()
            row2 = dataframe.iloc[[idx2]].reset_index()
            randInt = np.random.randint(9)
            if 0 <= randInt <= val_split:
                valid_frames = [validation_data, row1, row2]
                validation_data = pd.concat(
                    valid_frames, axis=0, join='outer', ignore_index=False)
            if randInt >= val_split + 1:
                train_frames = [training_data, row1, row2]
                training_data = pd.concat(
                    train_frames, axis=0, join='outer', ignore_index=False)

        return training_data, validation_data

    def adjust_brightness(self, image, factor, slice):
        # Convert to hue, saturation, value model
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv[:, :, slice] = hsv[:, :, slice] * factor
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    def crop_sky_and_dashboard(self, frame):
        frame = frame[100:440, :-90]
        image = cv2.resize(frame, (220, 66), interpolation=cv2.INTER_AREA)
        return image

    def optical_flow(self, im_c, im_n):
        gray_c = self.grayscale(im_c)
        gray_n = self.grayscale(im_n)
        hsv = np.zeros_like(im_c)
        hsv[:, :, 1] = cv2.cvtColor(im_n, cv2.COLOR_RGB2HSV)[:, :, 1]


        flow = cv2.calcOpticalFlowFarneback(gray_c, gray_n,
                                            flow_mat,
                                            image_scale,
                                            nb_images,
                                            win_size,
                                            nb_iterations,
                                            deg_expansion,
                                            STD,
                                            0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[:, :, 0] = ang * (180 / np.pi / 2)
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv = np.asarray(hsv, dtype=np.float32)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        #cv2.imwrite("./flow.jpg", rgb_flow)

        return rgb_flow


    def preprocess_image_valid_from_path(self, image_path, speed):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.adjust_brightness(img, 0.2, 2)    
        img = self.crop_sky_and_dashboard(img)
        return img, speed


    def preprocess_image_from_path(self, image_path, speed):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.adjust_brightness(img, 0.2, 2)    
        img = self.crop_sky_and_dashboard(img)
        return img, speed

if __name__ == "__main__":
    klass = PreProcessor()
    klass.plot_training_speed(LABELS)
    klass.generate_images(VIDEO_FILE, LABELS)
   # df = pd.read_csv('./processed.csv', header=None)
   # train, valid = klass.shuffle_frame_pairs(df)
   # im = cv2.imread(train[0].values[190])
   # im1 = cv2.imread(train[0].values[191])
   # im = klass.adjust_brightness(im, 1.6, 2)
   # im1 = klass.adjust_brightness(im1, 1.6, 2)
   # crop1 = klass.crop_sky_and_dashboard(im)
   # crop2 = klass.crop_sky_and_dashboard(im1)
   # optical = klass.optical_flow(crop1, crop2)
   # plt.imshow((optical * 255).astype(np.uint8))
   # plt.show()
   # print("Preprocessor")
