#Pytorch imports
import torch
import torchvision 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
#Misc Imports
import os , sys , time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import *



torch.backends.cudnn.benchmark = True

#List of Diseases that can be classfied from model
classes_name = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
n_classes = 14

class DenseNet(nn.Module):
        '''   Class for Adding a sigmoid layer in the denseNet Model.
        Args: 
            n_classes :(int) Number of classes
        '''
        def __init__(self,n_classes):
            super(DenseNet,self).__init__()
            self.densenet121 = models.densenet121(pretrained=True)
            n_filters = self.densenet121.classifier.in_features
            self.densenet121.classifier = nn.Sequential(nn.Linear(n_filters, n_classes), nn.Sigmoid())
            
        def forward(self, x):
            x = self.densenet121(x)
            return x


model = DenseNet(n_classes).cuda()
model = nn.DataParallel(model).cuda()    
checkpoint_path = r"D:\Projects\RDD\checkpoints\model.pth.rar"        
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])    
abc = model.module.densenet121.features  
model.eval()

def backends(data_dir):

    checkpoint_path = "model.pth.rar" 
    '''image_folder_path = './Datasets/pytorch-django-chexnet-master/chestx-ray-data/images'
    label_folder_path = './Datasets/txt_files'
    train_path = './Datasets/txt_files/train.txt'
    test_path = './test.txt'
    val_path = './Datasets/a/00011575_005.png'
    
    n_epochs = 60
    '''
    
        
    
    
    
    
    
    
    
    
    def save_checkpoint(model, best_acc, epochs, optimizer):
        ''' This function is used to save the checkpoint of the models.
        Args:
            model:(pytorch model) A torch Model that is to be saved
            best_acc:(float) The best accuracy attained during the training process
            epochs:(int) The number of epochs completed
            lr:(float) Current Learning rate of the model
        '''
        print('<------Saving Checkpoint----->')
        state = {'epoch': epochs,
                 'state_dict': model.state_dict(),
                 'best_loss': best_acc,
                 'optimizer' : optimizer.state_dict()}
        
        torch.save(state, checkpoint_path)
        print('<----Checkpoint saved------>')
    
        
    
    
    
    class DataPreprocessing(torch.utils.data.Dataset):
        def __init__(self, data_dir, transform=None):
            """
            Args:
                data_dir: path to image directory.
                image_list: path to the file containing images
                    with corresponding labels.
                transform: optional transform to be applied on a sample.
            """
            self.transform = transform
    
            
    
        def __getitem__(self, index):
            """
            Args:
                index: the index of item
    
            Returns:
                image and its labels
            """
            
            image = Image.open(data_dir).convert('RGB')
            
            if self.transform is not None:
                image = self.transform(image)
                
            return image
    
        def __len__(self):
            return 1
    
    
    '''
    def train():
        transform_list=[]
        normalize_data = ([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])  # Standard ImageNet Normalization Data
        
        transform_list.append(transforms.RandomResizedCrop(224))
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        transform_list.append(normalize_data)      
        transform_sequence=transforms.Compose(transform_list)
        
      
        #Loading data
        train_data = DataPreprocessing(img_path = image_folder_path,
                                       txt_path = train_path,
                                       transform = transform_sequence)
        test_data = DataPreprocessing(img_path = image_folder_path,
                                       txt_path = test_path,
                                       transform = transform_sequence)
        val_data = DataPreprocessing(img_path = image_folder_path,
                                       txt_path = val_path,
                                       transform = transform_sequence)
        
        train_dataLoader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=24, pin_memory=True)
        test_dataLoader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=24, pin_memory=True)
        val_dataLoader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=24, pin_memory=True)
        
        #Optimizer, Scheduler and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9 , 0.999), weight_decay = 1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
        criterion = nn.BCELoss(size_average = True)
    
        #Loading Checkpoint
        if os.path.isfile(checkpoint_path):
            print('Checkpoint Found.\n <---------------Loading Checkpoint------------>')
            chkpnt = torch.load(checkpoint_path)
            model.load_state_dict(chkpnt['state_dict'])
            optimizer.load_state_dict(chkpnt['optimizer'])
            model.eval()
        
        #--------------------Training Process of Model Starts from Here----------------
        min_loss = 99999.0
        running_loss_history = []
        val_loss_history = []
        for epoch in range(0, n_epochs):
            running_loss = 0.0
            val_running_loss = 0.0
            
            for i, (image, label) in tqdm(enumerate(train_dataLoader)):
                label = label.cuda()
                varIn = Variable(image)
                varLabel = Variable(label)
               
                varOut = model(varIn)
                
                loss = criterion(varOut, varLabel)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            epoch_loss = running_loss/len(train_dataLoader)
            running_loss_history.append(epoch_loss)
            val_loss, sch_loss = validation()
            
            scheduler.step(sch_loss.data[0])
            #Checking if validation loss is less than minimum loss. If yes, then checkpointing the model.
            if val_loss < min_loss:
                min_loss = val_loss
                save_checkpoint(model = model, best_acc = min_loss, epochs=epoch, optimizer = optimizer)
        print("Epoch {} Completed".format(epoch))
        print("Val Loss: " + min_loss)
                
            
            
            
    
    def validation():
        loss_mean = 0
        for i, (image,label) in enumerate(val_dataLoader):
            label = label.cuda()
            varIn = Variable(image, volatile= True)
            varLabel = Variable(label, volatile= True)
               
            varOut = model(varIn)
                
            loss = criterion(varOut, varLabel)
            loss_mean += loss
            val_running_loss += loss.item()
            
        val_epoch_loss = val_running_loss/len(val_dataLoader)
        loss_mean = loss_mean/len(val_dataLoader)
        val_loss_history.append(val_epoch_loss)
        return val_epoch_loss, loss_mean
        
    
    
    
    
    gt = torch.FloatTensor()
    gt=gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    '''
    
    '''   
    model = DenseNet(n_classes).cuda()
    model = nn.DataParallel(model).cuda()    
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])    
    abc = model.module.densenet121.features  
    model.eval()
        '''
        
           
            
    
    
    pred = torch.FloatTensor()
    pred = pred.cuda()
       
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        #Loading data
    test_data = DataPreprocessing(data_dir = data_dir,
                                       
                                       transform = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.TenCrop(224),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([normalize(crop) for crop in crops]))]
    ))
        
        
        
    test_dataLoader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        
      
    for i, (inp) in enumerate(test_dataLoader):
            bs, n_crops, c, h, w = inp.size()
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
            
            output = model(input_var)
            
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)
            pred = pred.double().cpu().numpy()[0]
    
    ret = {}
    i = 0
    for class_name in classes_name:
        ret.update({
                            class_name: round((pred[i]*100),2)
                        })
                        # ret[class_name] = pred[i]
        i += 1
    return ret        

def cam(data_dir):
                
                abc.eval()
                transformList = []
                normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                transformList.append(transforms.Resize(448))
                transformList.append(transforms.ToTensor())
                transformList.append(normalize)
                
                transeq = transforms.Compose(transformList)
               
                imageData = Image.open(data_dir).convert('RGB')
                imageData = transeq(imageData)
                imageData = imageData.unsqueeze_(0)
                
                inputd = torch.autograd.Variable(imageData)
                otpt = abc(inputd.cuda())
                
                heatmap=None
                weights = list(abc.parameters())[-2]
                for i in range(0,len(weights)):
                    maps = otpt[0,i]
                    if i==0:
                        heatmap = weights[i]*maps
                    else:
                        heatmap += weights[i]*maps
                npHeatmap = heatmap.cpu().data.numpy()
                imgOriginal = cv2.imread(data_dir, 1)
    
                imgOriginal = cv2.resize(imgOriginal, (448, 448))
                
                cam = npHeatmap / np.max(npHeatmap)
                cam = cv2.resize(cam, (448, 448))
                heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
                          
                img = heatmap * 0.5 + imgOriginal
                return img
                


     
        
    