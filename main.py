#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 09:53:52 2018

@author: kneehit
"""
 

 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import transforms
import os
#os.chdir('/home/kneehit/Data Science/Bone Age Kaggle/PyTorch')

from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import glob
import random
from age_predictor_model import Bottleneck, AgePredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# os.chdir('/home/kneehit/Data Science/Bone Age Kaggle/PyTorch')


#%%
# Image and CSV paths
train_dataset_path = 'bone_data/train/'
test_dataset_path = 'bone_data/test/'
val_dataset_path = 'bone_data/validation/'
csv_path = 'bone_data/boneage-training-dataset.csv'



# For reproducibility use the seeds below 
torch.manual_seed(1498920)
torch.cuda.manual_seed(1498920)
np.random.seed(1498920)
random.seed(1498920)
torch.backends.cudnn.deterministic=True

#%% 
# Sample random images from dataset (since loading all images will take time) 
# and calculate mean and standard deviation for normalization 
k = 100
size = 500
image_filenames = glob.glob(train_dataset_path+'*.png')
random_images = random.sample(population = image_filenames,k = k)


means = []
stds = []

for filename in random_images:
    image = cv2.imread(filename,0)
    image = cv2.resize(image,(size,size))
    mean,std = cv2.meanStdDev(image)
#    mean /= 255
#    std /= 255
    
    means.append(mean[0][0])
    stds.append(std[0][0])

avg_mean = np.mean(means) 
avg_std = np.mean(stds)

print('Approx. Mean of Images in Dataset: ',avg_mean)
print('Approx. Standard Deviation of Images in Dataset: ',avg_std)




# To reproduce results use below values
#avg_mean = 52.96
#avg_std = 26.19
#%%

# Split Train Validation Test
# Train - 10000 images
# Val   -  1611 images
# Test  -  1000 images

dataset_size = len(image_filenames)
val_size = dataset_size + 1611



bones_df = pd.read_csv(csv_path)
bones_df.iloc[:,1:3] = bones_df.iloc[:,1:3].astype(np.float)


train_df = bones_df.iloc[:dataset_size,:]
val_df = bones_df.iloc[dataset_size:val_size,:]
test_df = bones_df.iloc[val_size:,:]


age_max = np.max(bones_df['boneage'])
age_min = np.min(bones_df['boneage'])
#%%
class BonesDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):

        self.dataframe = dataframe

        
        self.image_dir = image_dir
        self.transform = transform
        

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_name = self.image_dir + str(self.dataframe.iloc[idx,0]) + '.png'
        image = cv2.imread(img_name,0)
        image = image.astype(np.float64)
        gender = np.atleast_1d(self.dataframe.iloc[idx,2])
        bone_age = np.atleast_1d(self.dataframe.iloc[idx,1])
        
        

        sample = {'image': image, 'gender': gender, 'bone_age':bone_age}

        if self.transform:
            sample = self.transform(sample)

        return sample

#%% 
# Custom Transforms for Image and numerical data
        
# Resize and Convert numpy array to tensor
class ToTensor(object):
    

    def __call__(self, sample):
        image, gender, bone_age = sample['image'], sample['gender'], sample['bone_age']


        image = cv2.resize(image,(size,size))
        image = np.expand_dims(image,axis = 0)
        
#       we need to convert  cuda.longtensors to cuda.floatTensor data type
        return {'image': torch.from_numpy(image).float(),
                'gender': torch.from_numpy(gender).float(),
                'bone_age':torch.from_numpy(bone_age).float()}        

# Normalize images and bone age
class Normalize(object):
    
    def __init__(self,img_mean,img_std,age_min,age_max):
        self.mean = mean
        self.std = std
        
        self.age_min = age_min
        self.age_max = age_max
        
    
    
    def __call__(self,sample):
        image, gender, bone_age = sample['image'], sample['gender'], sample['bone_age']
        
        image -= self.mean
        image /= self.std
        
        bone_age = (bone_age - self.age_min)/ (self.age_max - self.age_min)
        
        
        
        return {'image': image,
                'gender': gender,
                'bone_age':bone_age} 
        


data_transform = transforms.Compose([
   Normalize(avg_mean,avg_std,age_min,age_max),
   ToTensor()
   
   ])     
    


#%%
train_dataset = BonesDataset(dataframe = train_df,image_dir=train_dataset_path,transform = data_transform)
val_dataset = BonesDataset(dataframe = val_df,image_dir = val_dataset_path,transform = data_transform)
test_dataset = BonesDataset(dataframe = test_df,image_dir=test_dataset_path,transform = data_transform)

# Sanity Check
print(train_dataset[0])

     
train_data_loader = DataLoader(train_dataset,batch_size=4,shuffle=False,num_workers = 4)
val_data_loader = DataLoader(val_dataset,batch_size=4,shuffle=False,num_workers = 4)
test_data_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers = 4)

#%%
   

# Sanity Check 2
sample_batch =  next(iter(test_data_loader))
print(sample_batch)

#%%
# Initialize the model
age_predictor = AgePredictor(block = Bottleneck,layers = [3, 4, 23, 3],num_classes =1)


#%%
# Set loss as mean squared error (for continuous output)
# Initialize Stochastic Gradient Descent optimizer and learning rate scheduler

age_predictor = age_predictor.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(age_predictor.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.5)




#%% To Resume Training 


#checkpoint = torch.load('epoch-25-loss-0.0194-val_loss-0.0085.pth.tar')
#start_epoch = checkpoint['epoch']
#age_predictor.load_state_dict(checkpoint['state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer'])
#scheduler.load_state_dict(checkpoint['scheduler'])



#%%

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def eval_model(model,data_loader):
    model.eval()

    with torch.no_grad():
        
        result_array = np.array([])
        
        for batch_no,batch in enumerate(data_loader):
            
            optimizer.zero_grad()
            
            image = batch['image'].to(device)
            gender = batch['gender'].to(device)
            
    
            outputs = model(image,gender)
            preds = outputs.cpu().numpy()
    
            preds = preds.reshape(preds.shape[0])
            preds = denormalize(preds,age_min,age_max)
            
            result_array = np.concatenate((result_array,preds))
            
        return result_array
    


# Training Loop
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    
    for epoch in range(num_epochs):
        scheduler.step()
        model.train()
        running_loss = 0.0
        val_running_loss = 0.0
        
        for batch_no,batch in enumerate(train_data_loader):
            # Load batch
            image = batch['image'].to(device)
            gender = batch['gender'].to(device)
            age = batch['bone_age'].to(device)

            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                # Forward Pass
                outputs = model(image,gender)
                loss = criterion(outputs, age)
                
                # Backprop
                loss.backward()
                optimizer.step()
                
            # Calculate Loss
            running_loss += loss.item() * image.size(0)
            
            if (batch_no + 1) % 25 == 0: print('Epoch {} Batch {}/2500 '.format(epoch+1,batch_no+1))
            
        total_loss = running_loss / dataset_size
        print('\n \n Epoch {} Loss: {:.4f} \n \n'.format(epoch+1,total_loss))
        
        
    # Eval on validation set
    model.eval()
    for val_batch in val_data_loader:
        image = val_batch['image'].to(device)
        gender = val_batch['gender'].to(device)
        age = val_batch['bone_age'].to(device)

        
        optimizer.zero_grad()  
        # only forward pass, dont update gradients
        with torch.set_grad_enabled(False):
            outputs = model(image,gender)
            loss = criterion(outputs, age)
            
        val_running_loss += loss.item() * image.size(0)
    
    val_loss = val_running_loss / 1611
    
    print('Validation Loss {:.4f}'.format(val_loss))
        
        
        
        
    # Save checkpoint every epoch
    total_epochs = scheduler.state_dict()['last_epoch'] + 1        
    states = {
            'epoch': total_epochs + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler'  : scheduler.state_dict()
        }
    save_checkpoint(states,filename = 'epoch-{}-loss-{:.4f}-val_loss-{:.4f}.pth.tar'.format(total_epochs,total_loss,val_loss))

    return model
#%%

resnet_model = train_model(age_predictor,criterion,optimizer,scheduler,num_epochs=20)
#%%    

def denormalize(inputs,age_min,age_max):
    return inputs * (age_max - age_min) + age_min
        



result_array = eval_model(age_predictor,test_data_loader)

test_df['output'] = result_array
test_df['output'] = np.round(test_df['output'], decimals=2)
test_df = test_df.reset_index()

#%%
rmse = np.sqrt(mean_squared_error(test_df['boneage'], test_df['output']))
print(rmse)
# 25.259
#%%

def display_preds(num):
    idx = random.sample(range(0, test_df.shape[0]), num)
    for i in idx:
        image = cv2.imread(test_dataset_path + str(test_df['id'][i]) + '.png')
        image = cv2.resize(image,(500,500))
        cv2.putText(image,'Actual:' + str(test_df['boneage'][i]),(20,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255))
        cv2.putText(image,'Predicted:' + str(test_df['output'][i]),(20,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255))
        cv2.imshow('Bone Age Prediction',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
display_preds(4)

   






