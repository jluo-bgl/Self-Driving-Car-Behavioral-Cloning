# self-driving-car-behavioral-cloning
This is [UDacity](https://www.udacity.com/drive) Self Driving Car Behavioral Cloning Project

This repository arms to help me pass the project and helps you find all related data / method
in a single place.

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


#Iterations
###Self Struggling Car
1. Center Images
2. No Augmention
3. Nvidia Model with one dropout
4. 5 Apoch, Adam 0.001 learning rate
5. 55% validation accuracy
To reproduce this iteration, run below code
```python
    data_provider = DriveDataProvider(
        *DrivingDataLoader("datasets/udacity-sample-track-1/driving_log.csv").images_and_angles())
    data_provider.save_to_file("datasets/udacity-sample-track-1/driving_data.p")
```
```python
    data_provider = DriveDataProvider.load_from_file("datasets/udacity-sample-track-1/driving_data.p")
    Trainer(data_provider).fit()
```

