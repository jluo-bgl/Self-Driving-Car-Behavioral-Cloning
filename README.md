#Learn human driving behavior based on deep neural network
This is [UDacity](https://www.udacity.com/drive) Self Driving Car Behavioral Cloning Project

This repository arms to help me pass the project and helps you who is learning deep learning to
1. Easy to experiment, from simply apply CNN model to very complex data augment
2. Visualise what's going on
3. Build more understanding about how deep learning works

###Before Start
####Existing Solutions
#####NVIDIA
Nvidia has published a nice paper [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
This video will makes you very exciting.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=-96BEoXJMs0
" target="_blank"><img src="http://img.youtube.com/vi/-96BEoXJMs0/0.jpg" 
alt="NVIDIA AI Car Demonstration" width="400" height="360" border="10" /></a>

#####Commaai
[The Paper](https://arxiv.org/abs/1608.01230)
[Github Repository](https://github.com/commaai/research)
[train_steering_model.py](https://github.com/commaai/research/blob/master/train_steering_model.py)

####Data Collection
1. [UDacity](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) 
provided a simulator and produced a sample data for track 1 that you can use.
**this is recommended way**
2. Generate your data from UDacity Self-Driving Car Simulator
3. [Sully Chen](https://github.com/SullyChen) 
had a [TensorFlow implementation](https://github.com/SullyChen/Autopilot-TensorFlow) 
and shared his own [dataset](https://drive.google.com/file/d/0B-KJCaaF7ellQUkzdkpsQkloenM/view?usp=sharing)

###Data Pre-processing
####Data Input Size
Nvidia: 3@66x200
Commaai: 3@160x320
Udacity: 3@160x320

### Training

### Testing

####Simulator
```bash
python drive.py model.json
```
this script will read model.json and model.h5, and play UDacity in Autonomous Mode

#Architecture
The whole system has been designed for easy to 
1. Experiment
2. Understand
3. Extend

##Data

##Model

#Iterations
###Iteration 1 Self Stuck Car
1. Center Images
2. No Augmention
3. Nvidia Model with one dropout
4. 5 Apoch, Adam 0.001 learning rate
5. 55% validation accuracy
To reproduce this iteration, run below code
```python
dataset = DriveDataSet("datasets/udacity-sample-track-1/driving_log.csv")
data_generator = DataGenerator(center_image_generator)
Trainer(learning_rate=0.0001, epoch=10).fit(data_generator.generate(dataset, batch_size=128))
```
<a href="http://www.youtube.com/watch?feature=player_embedded&v=mmGoI1crA9s" target="_blank">
<img src="http://img.youtube.com/vi/mmGoI1crA9s/0.jpg" alt="Iteration 1 Self Stuck Car" width="400" height="360" border="10" /></a>

###Iteration 2 Center/Left/Right Images, able to make first turn
As i'm running into 2GB file saving issue in python, it's time to start involve in Keras generator
so that I don't need create a super large file and load it into memory
```python
dataset = DriveDataSet("datasets/udacity-sample-track-1/driving_log.csv")
data_generator = DataGenerator(center_left_right_image_generator)
Trainer(learning_rate=0.0001, epoch=10).fit(data_generator.generate(dataset, batch_size=128))
```
<a href="http://www.youtube.com/watch?feature=player_embedded&v=NlQLqaX0qqE" target="_blank">
<img src="http://img.youtube.com/vi/NlQLqaX0qqE/0.jpg" alt="Iteration 2 First Turn Succeed" width="400" height="360" border="10" /></a>


###Iteration 3 Shift Image Randomly
so far we have made use of all provided data, other idea is that shift the images and adjust angles accordingly.
for example, center image with angle 0, move 10 pixels left would result angle 0.04
<image1><image2><image3>
In this iteration, we are facing a slow generator issue, as some images have to shift in the runtime,
the training time for on epoch need 10 minutes, where it's take 50 seconds in iteration 2.
The car able to reach the bridge as the best score.

```python
dataset = DriveDataSet("datasets/udacity-sample-track-1/driving_log.csv", crop_images=False)
data_generator = DataGenerator(
    random_generators(
        random_center_left_right_image_generator,
        pipe_line_generators(
            random_center_left_right_image_generator,
            shift_image_generator
        )
    ))
Trainer(learning_rate=0.0001, epoch=10, multi_process=True).fit(
    data_generator.generate(dataset, batch_size=128),
    input_shape=dataset.output_shape()
)
```

###Iteration 4, Crop to 66x200
To improve the performance, we start to support crop, images will reduced from 160x320 to 66x200
this modification change the whole framework to support different input data shape, this include
1. drive.py support any input shape
2. Trainer will ask DriveDataSet for input shape and pass into model

>The crop and multi_process reduced training time from 10 minutes to 1.5 minutes
>please note, the whole system still support full image to been trained, just added one more parameter
while contruct DriveDataSet(crop_images=True)

Surprisely, without any change in system, our car able to drive a whole lap now.
also trainable params dropped from **32,213,367** to **1,595,511**,
the weight file size dropped from **128m** to **6.4m**,
it's a huge train time save.

```python
dataset = DriveDataSet("datasets/udacity-sample-track-1/driving_log.csv", crop_images=True)
data_generator = DataGenerator(
    random_generators(
        random_center_left_right_image_generator,
        pipe_line_generators(
            random_center_left_right_image_generator,
            shift_image_generator
        )
    ))
Trainer(learning_rate=0.0001, epoch=10, multi_process=True).fit(
    data_generator.generate(dataset, batch_size=128),
    input_shape=dataset.output_shape()
)
```
