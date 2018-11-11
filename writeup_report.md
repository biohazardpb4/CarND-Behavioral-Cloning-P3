# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./images/model.png "model"
[center]: ./images/center.jpg "center"
[post_flip]: ./images/post_flip.jpg "post_flip"
[pre_flip]: ./images/pre_flip.jpg "pre_flip"
[recover_center]: ./images/recover_center.jpg "recover_center"
[recover_left]: ./images/recover_left.jpg "recover_left"
[recover_right]: ./images/recover_right.jpg "recover_right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* models/with_slow_bridge.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py models/with_slow_bridge.h5
```

#### 3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 layers of 5x5 filters and 2 layers of 3x3 filters. It has depths between 36 and 128 (train.py lines 55-59) 

The model includes RELU layers to introduce nonlinearity (code lines 55-59), and the data is normalized in the model using a Keras lambda layer (code line 53). The imagery data is also cropped on line 54 to avoid surroundings above the horizon influencing driving.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 68). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (train.py line 66).

#### 4. Appropriate training data
Training data was chosen to keep the vehicle driving on the road. I used a combination of four center laps, one lap with smooth turns, a couple runs with recovery from the sides, a counter clockwise lap, bridge-specific navigation, and turning around the last turn.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to design an architecture appropriate for taking car camera images as input and producing a steering angle as output. Convolutional nets are good at recognizing patterns in images, so those were used.

My first step was to use a convolution neural network model similar to [Nvidia's E2E Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) model. I thought this model might be appropriate because it was suggested as part of the project materials. It also appears appropriate because it's solving a the same problem, although ours is limited to only using cameras and only controlling steering, as opposed to using other sensors like radar and controlling things like acceleration.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on the training set but a low mean squared error on the validation set. According to what we had learned so far, this was very odd. I gathered some extra data, replacing my keyboard control with mouse control since I figured having the continuous mouse data vs the on/off keyboard data would be more valuable to training. After looking around online, I came to the conclusion that maybe the model did not have enough parameters, so I expanded its size. After expanding the size of the network, it came to a combination of errors that was recognizable as overfitting, with the training set having lower MSE than the validation set. 

To combat the overfitting, I attempted inserting drop out layers in a few different places, and increased the learning rate of the optimizer as suggested in dropout tips that I saw online.

The dropout was ultimately discarded in favor of gathering more data around specific problem points in the track and training for a fewer number of epochs to reduce overfitting.

While iterating on the model, the simulator was run after training to see how well the car was driving around track one. Whenever the vehicle had trouble getting past a specific spot, I would use the simulator to gather new data in training mode and train a further model using the same architecture.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (train.py lines 52-64) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it fell out of the center of the lane. These images show what a recovery looks like starting from left, center, and right:

![alt text][recover_left]
![alt text][recover_center]
![alt text][recover_right]

To augment the data sat, I also flipped images and angles thinking that this would ensure that the model does not only know how to turn left like a NASCAR car. For example, here is an image that has then been flipped:

![alt text][pre_flip]
![alt text][post_flip]

After the collection process, I had 38968 of data points. I then augmented this data via flipping horizontally. As part of the model, the data is normalized to have a mean closer to 0 and also cropped to the portion of the image which typically includes the road, so that parameters are not wasted on taking surroundings into account.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the error in the validation tending to cease to decrease after the 7th epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here's a [link to my video result](./recording/output_video.mp4).
