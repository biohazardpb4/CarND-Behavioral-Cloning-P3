import csv
import cv2
import numpy as np
from os import listdir
from os.path import join

INCLUDE_LR=False

augmented_images, augmented_measurements = [], []
collected_data_path = './collected_data/'
for run in listdir(collected_data_path):
    run_path = join(collected_data_path, run)
    print('ingesting run: {}'.format(run_path))
    lines = []
    with open(join(run_path, 'driving_log.csv')) as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)

    images, measurements = [], []
    for line in lines:
        center_left_right = line[0:2]
        filenames = [source_path.split('/')[-1] for source_path in center_left_right]
        impaths = [join(run_path, 'IMG', filename) for filename in filenames]
        measurement = float(line[3])
        if INCLUDE_LR:
            images.extend([cv2.imread(impath) for impath in impaths])
            # derived from multi-camera example image
            correction = 0.15
            left_measurement = measurement + correction
            right_measurement = measurement - correction
            measurements.extend([measurement, left_measurement, right_measurement])
        else:
            images.append(cv2.imread(impaths[0]))
            measurements.append(measurement)
        
    # Augment images by flipping horizontally and negating steering angle.
    for image, measurement in zip(images, measurements):
      augmented_images.append(image)
      augmented_measurements.append(measurement)
      augmented_images.append(np.fliplr(image))
      augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(60, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(200))
model.add(Dense(70))
model.add(Dense(10))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=0.002))
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')

