import csv
import cv2
import numpy as np

lines = []
with open('./collected_data/center_full_lap_counterclockwise/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

images, measurements = [], []
for line in lines:
    center_left_right = line[0:2]
    filenames = [source_path.split('/')[-1] for source_path in center_left_right]
    impaths = ['./collected_data/center_full_lap_counterclockwise/IMG/' + filename for filename in filenames]
    images.extend([cv2.imread(impath) for impath in impaths])
    measurement = float(line[3])
    # derived from multi-camera example image
    correction = 0.15
    left_measurement = measurement + correction
    right_measurement = measurement - correction
    measurements.extend([measurement, left_measurement, right_measurement])

# Augment images by flipping horizontally and negating steering angle.
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(np.fliplr(image))
  augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')

