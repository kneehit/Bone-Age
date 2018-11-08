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
Predictions on the test set can be seen below

![alt text](images/pred_on_test.gif)
