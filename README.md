#Learn human driving behavior based on deep neural network
This is [UDacity](https://www.udacity.com/drive) Self Driving Car Behavioral Cloning Project

Lots of blog / repositories in internet just show you their final result, but how did they reach their beautiful 
final result is really the most important part for a learner point of view.

This repository arms to help me as a newbie and helps you who is learning deep learning to
1. Easy to experiment, from simply apply CNN model to very complex data augment
2, Reproducible, every bad result we keep it reproducible so that we know we made a mistake buy what reason
3. Visualise what's going on
4. Build more understanding about how deep learning works

To help achieve above goal, all code base has been formed by below layers or pipes

| Layer             | Purpose                                                                                           |
| ------------------|---------------------------------------------------------------------------------------------------|
| DriveDataSet      | Represent the data you recorded                                                                   |
|   filter_method   | What data you'd like to added in                                                                  |
| RecordAllocator   | Before pass recorded data to data augment, percentage of different data you'd like to added in    |
| generators        | Data augment process you'd like to apply to, easy to extend to any order                          |
| DataGenerator     | Read from RecordAllocator, pass to generator, then feed data into Keras generator                 |
| model             | the Network                                                                                      |
| Trainer           | create Model, read data from DataGenerator, do the real training                                  |


When we put everything together, the code looks like:
```python
data_set = DriveDataSet.from_csv(
    "datasets/udacity-sample-track-1/driving_log.csv", crop_images=True,
    filter_method=drive_record_filter_include_all)

allocator = AngleTypeWithZeroRecordAllocator(
    data_set, left_percentage=20, right_percentage=20,
    zero_percentage=8, zero_left_percentage=6, zero_right_percentage=6,
    left_right_image_offset_angle=0.25)
generator = pipe_line_generators(
    shift_image_generator(angle_offset_pre_pixel=0.002),
    flip_generator,
    brightness_image_generator(0.25)
)
data_generator = DataGenerator(allocator.allocate, generator)
Trainer(learning_rate=0.0001, epoch=10, dropout=0.5).fit(
    data_generator.generate(batch_size=128),
    input_shape=data_set.output_shape()
)
```

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
<img src="http://img.youtube.com/vi/mmGoI1crA9s/0.jpg" alt="Iteration 1 Self Stuck Car" width="600" height="360" border="10" /></a>

###Iteration 2 Center/Left/Right Images, able to make first turn
As i'm running into 2GB file saving issue in python, it's time to start involve in Keras generator
so that I don't need create a super large file and load it into memory
```python
dataset = DriveDataSet("datasets/udacity-sample-track-1/driving_log.csv")
data_generator = DataGenerator(center_left_right_image_generator)
Trainer(learning_rate=0.0001, epoch=10).fit(data_generator.generate(dataset, batch_size=128))
```
<a href="http://www.youtube.com/watch?feature=player_embedded&v=NlQLqaX0qqE" target="_blank">
<img src="http://img.youtube.com/vi/NlQLqaX0qqE/0.jpg" alt="Iteration 2 First Turn Succeed" width="600" height="360" border="10" /></a>


###Iteration 3 Shift Image Randomly
so far we have made use of all provided data, other idea is that shift the images and adjust angles accordingly.
for example, center image with angle 0, move 10 pixels left would result angle 0.04
<image1><image2><image3>
In this iteration, we are facing a slow generator issue, as some images have to shift in the runtime,
the training time for on epoch need 10 minutes, where it's take 50 seconds in iteration 2.
I discontinued this iteration and continue to 4

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

also trainable params dropped from **32,213,367** to **1,595,511**,
the weight file size dropped from **128m** to **6.4m**,
it's a huge train time save.

In this iteration, we are able to drive until bridge, occasionally it's able to drive whole lap.

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


###Iteration 5, remove shift from left and right camera images
while look at the video, I noticed that it's go wild when approch road side, that's must be something wrong 
with shift image, the angle we offset may too much, the left/right image we are using may use different ratio.
so that in this test, just simple remove left right image before shift.
As you can see in video, the car is much more smooth and can drive longer.
```python
data_generator = DataGenerator(
    random_generators(
        center_image_generator,
        pipe_line_generators(
            random_center_left_right_image_generator,
            shift_image_generator
        )
    ))
```


###Iteration 6 Flip Image
```python
data_generator = DataGenerator(
    random_generators(
        random_center_left_right_image_generator,
        pipe_line_generators(
            center_image_generator,
            shift_image_generator
        ),
        pipe_line_generators(
            random_center_left_right_image_generator,
            flip_generator
        )
    ))
```

###Iteration 7 The dropout to rescue
0.5 Droput in every lay improved the performance much better


###Iteration 8 Convert generator to batch model and convert into Tensorflow 
after add flip image, it took 2 minutes for every epoch, which is still too long for me.
most of the time, my GPU is waiting for image to been generated. 
convert the augment into tensorflow to do it in batch, gpu and true multiple thread


###Iteration 9 Feeding data distribution
It looks we running out of option for augment our data, one way is go back to simulator and generate more data,
also we know the model should work, the issue must be in the data, either not enough or we baies the model too much,
in track 1, car is turning left far more than right, maybe that's why our car not able to handle the turning right 
very well.
let's look back and see what kind of data we feed into model.
The Udacity Sample data has below distribution
<image>
As you can see, angle 0 (going straight) has far more samples, as we used left and right camera data, 0.25 and -0.25
is same.
what it happened in real world of our steering angle distributed? I guess maybe it's 25% of left and right turn, 50% 
of straight.