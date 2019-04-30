# Bone Age Predictor

![alt text](images/flowchart.png)

## Description
The aim of this project is to predict Bone Age (in months) from the X-Ray and Gender.<br>
Dataset - https://www.kaggle.com/kmader/rsna-bone-age

## Motivation
I chose this project primarily since it combines more than one form of data and simply using pretrained model won't work in this scenario.

Following are the challenges I encountered (and overcame) in this project:
1. Image data as well as categorical data is provided and hence will require a custom model.
2. Unlike most image recognition problem, the target variable is continuous and the input image has only single channel.
3. Since the ouput is continuous, the network can predict negative values (whereas age can only be positive).


## Results
Predictions (in number of months) on the test dataset can be seen below

![alt text](images/pred_on_test.gif)

## Closing Thoughts
Although the predictions are decent, there is certainly room for improvement. 
Following things can be experimented: <br>
1. Changing the architecture <br>
  a. Adding more layers - convolutional/FC/ResNet blocks <br>
  b. Replacing ResNet layers by layers of other CNN architecture like Inception, ResNeXt, etc. <br>
2. Using features from convolutional layers of a CNN which is pretrained on X-Rays of hands. 


