import numpy as np
import os
import sys
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as init
from model.dense_ed import DenseED
import random
import pandas as pd
#import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#matplotlib inline

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

#import pickle
import pickle
import warnings
import os
import json
from pprint import pprint
import scipy.io as io
import PIL #added PIL
from PIL import Image
import pandas
from skimage.transform import resize

from torch.utils import data

#from other file
import os
import numpy as np
from PIL import Image
import numbers
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor, _is_pil_image
import sys
# import Augmentor
from skimage.util.noise import random_noise
import scipy.misc
plt.switch_backend('agg')
import torch
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cpu:0'))

device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

net = 1 #change to 0 for u-net with bach norm, 1: DnCNN

test_type = 1
CSV_path = '/afs/crc.nd.edu/user/v/vmannam/Desktop/Summer20/May20/1905/Pytorch_training/'
class DataGenerator(data.Dataset):
    def __init__(self, num_exps=100, batch_size=4, dim=(128,128), n_channels=1, shuffle=True, train = True, validation = False, test = False, test_type=1, transform=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.num_exps = num_exps
        self.train = train
        self.validation = validation
        self.test = test
        self.on_epoch_end()
        self.test_type = test_type
        self.transform = transform

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_exps / self.batch_size))

    def __getitem__(self, index): #index = batch num
        'Generate one batch of data'
        # Generate indexes of the batch
        if self.train == False and self.validation == False and self.test == True:
            in1 = np.array([0])
            in2 = in1.tolist()
            indexes = in2
            #print('ind here: ',indexes)
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = indexes#[self.list_IDs[k] for k in indexes]
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        #if self.transform:
        #    X, Y = ToTensor()

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_exps)
        if self.train == True:
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty(((self.batch_size)*4, self.n_channels, *self.dim)) #total  batch size here includes slices
        Y = np.empty(((self.batch_size)*4, self.n_channels, *self.dim))
        if self.train == True and self.validation == False and self.test == False:
            df = pandas.read_csv(CSV_path + 'samples_train_all.csv')
        if self.train == False and self.validation == True and self.test == False:
            df = pandas.read_csv(CSV_path + 'samples_test.csv')
        if self.train == False and self.validation == False and self.test == True:
            if self.test_type == 1:
                df = pandas.read_csv(CSV_path + 'samples_test_inference_short.csv')
            if self.test_type == 2:
                df = pandas.read_csv(CSV_path + 'samples_test_inference2.csv')
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            x_img = np.asarray(PIL.Image.open(df['Images'][ID]))
            y_label = np.asarray(PIL.Image.open(df['Labels'][ID]))
            #print('X values', x_img[100:120,100:120])#[0,100:120,100:120])
            #print('train file name >>>>>>', df['Images'][ID])
            
            x_img = resize(x_img, (256, 256), mode='constant', preserve_range=True) #actual  image and label
            y_label = resize(y_label, (256, 256), mode='constant', preserve_range=True)#actual  image and label
            
            #slices here of 128x128
            imx1 = np.array([x_img[x:x+self.dim[0],y:y+self.dim[1]] for x in range(0,x_img.shape[0],self.dim[0]) for y in range(0,x_img.shape[1],self.dim[1])])
            imx1 = np.array(imx1)
            #print('>>>> ',imx1.shape)
            lbx1 = np.array([y_label[x:x+self.dim[0],y:y+self.dim[1]] for x in range(0,y_label.shape[0],self.dim[0]) for y in range(0,y_label.shape[1],self.dim[1])])
            #print('>>>>>>>>>>>>>>> ',X.shape)
            
            X[i*4:(i+1)*4, 0, ...] = imx1 / 65535 - 0.5
            #print('XXXXXX ',X.shape)
            Y[i*4:(i+1)*4, 0, ...] = lbx1 / 65535 - 0.5    #.squeeze()

        return torch.from_numpy(X), torch.from_numpy(Y)
        
#adding this class to convert numpy images in to the Tensors        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        noisy_image, clean_image = sample['X'], sample['Y']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(noisy_image),
                'landmarks': torch.from_numpy(clean_image)}        

batch_size = 4
batch_size_test = 1
im_height = 128 #M
im_width = 128  #N
num_exps_train = 400 #total train examples = 57000 we are using all FOV 9 to 16 here which is 8*50 = 400 images
num_exps_validation = 40  #total test examples = 3000
num_exps_test = 1 #test dataset with raw data #1 FOV 8 image
        
params_train = {'dim': (im_height,im_width),'num_exps':num_exps_train,'batch_size': batch_size,'n_channels': 1,'shuffle': True, 'train': True, 'validation': False, 'test': False, 'test_type':test_type, 'transform': ToTensor}
#params_validation= {'dim': (im_height,im_width),'num_exps':num_exps_validation,'batch_size': batch_size,'n_channels': 1,'shuffle': False,'train': False, 'validation': True, 'test': False, 'test_type':test_type, 'transform': ToTensor}
        #test_set =1 raw data, 2 -> avg2 , 3-> avg4, 4 -> avg8, 5-> avg16, 6-> all together noise levels (1,2,4,8,16)
params_test = {'dim': (im_height,im_width),'num_exps':num_exps_test,'batch_size': batch_size_test,'n_channels': 1,'shuffle': False, 'train': False,  'validation': False, 'test': True, 'test_type':test_type, 'transform': ToTensor}

# training_generator = DataGenerator( **params_train)
# validation_generator = DataGenerator(**params_validation)
# test_generator = DataGenerator(**params_test)

transformed_dataset_train = DataGenerator(**params_train) #convert to tensor 
dataloader_train = DataLoader(transformed_dataset_train, batch_size=1, shuffle=True, num_workers=4) #conver to data loader

#train_loader = training_generator
#print('len of train loader',len(dataloader_train))
#print('shape of train loader',dataloader_train)
#test_loader = test_generator
transformed_dataset_test = DataGenerator(**params_test) #convert to tensor 
dataloader_test = DataLoader(transformed_dataset_test, batch_size=1, shuffle=False, num_workers=4) #conver to data loader
#print('len of test loader',len(dataloader_test))
#from here onwards the model is added
import torch
import torch.nn as nn
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
class UpsamplingNearest2d(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

#models definitions here
class UnetN2N(nn.Module): #this is with the batch-norm version
    """
    Lehtinen, Jaakko, et al. "Noise2Noise: Learning Image Restoration without 
    Clean Data." arXiv preprint arXiv:1803.04189 (2018).
    Add BatchNorm and Tanh out activation
    """
    def __init__(self, in_channels, out_channels):
        super(UnetN2N, self).__init__()

        self.enc_conv0 = conv3x3(in_channels, 48)
        self.enc_relu0 = nn.LeakyReLU(0.1)
        self.enc_conv1 = conv3x3(48, 48)
        self.enc_bn1 = nn.BatchNorm2d(48)
        self.enc_relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 128
        self.enc_conv2 = conv3x3(48, 48)
        self.enc_bn2 = nn.BatchNorm2d(48)
        self.enc_relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 64
        self.enc_conv3 = conv3x3(48, 48)
        self.enc_bn3 = nn.BatchNorm2d(48)
        self.enc_relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # 32
        self.enc_conv4 = conv3x3(48, 48)
        self.enc_bn4 = nn.BatchNorm2d(48)
        self.enc_relu4 = nn.LeakyReLU(0.1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        # 16
        self.enc_conv5 = conv3x3(48, 48)
        self.enc_bn5 = nn.BatchNorm2d(48)
        self.enc_relu5 = nn.LeakyReLU(0.1)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        # 8
        self.enc_conv6 = conv3x3(48, 48)
        self.enc_bn6 = nn.BatchNorm2d(48)
        self.enc_relu6 = nn.LeakyReLU(0.1)
        self.upsample5 = UpsamplingNearest2d(scale_factor=2)
        # 16
        self.dec_conv5a = conv3x3(96, 96)
        self.dec_bn5a = nn.BatchNorm2d(96)
        self.dec_relu5a = nn.LeakyReLU(0.1)
        self.dec_conv5b = conv3x3(96, 96)
        self.dec_bn5b = nn.BatchNorm2d(96)
        self.dec_relu5b = nn.LeakyReLU(0.1)
        self.upsample4 = UpsamplingNearest2d(scale_factor=2)
        # 32
        self.dec_conv4a = conv3x3(144, 96)
        self.dec_bn4a = nn.BatchNorm2d(96)
        self.dec_relu4a = nn.LeakyReLU(0.1)
        self.dec_conv4b = conv3x3(96, 96)
        self.dec_bn4b = nn.BatchNorm2d(96)
        self.dec_relu4b = nn.LeakyReLU(0.1)
        self.upsample3 = UpsamplingNearest2d(scale_factor=2)
        # 64
        self.dec_conv3a = conv3x3(144, 96)
        self.dec_bn3a = nn.BatchNorm2d(96)
        self.dec_relu3a = nn.LeakyReLU(0.1)
        self.dec_conv3b = conv3x3(96, 96)
        self.dec_bn3b = nn.BatchNorm2d(96)
        self.dec_relu3b = nn.LeakyReLU(0.1)
        self.upsample2 = UpsamplingNearest2d(scale_factor=2)
        # 128
        self.dec_conv2a = conv3x3(144, 96)
        self.dec_bn2a = nn.BatchNorm2d(96)
        self.dec_relu2a = nn.LeakyReLU(0.1)
        self.dec_conv2b = conv3x3(96, 96)
        self.dec_bn2b = nn.BatchNorm2d(96)
        self.dec_relu2b = nn.LeakyReLU(0.1)
        self.upsample1 = UpsamplingNearest2d(scale_factor=2)
        # 256
        self.dec_conv1a = conv3x3(96 + in_channels, 64)
        self.dec_bn1a = nn.BatchNorm2d(64)
        self.dec_relu1a = nn.LeakyReLU(0.1)
        self.dec_conv1b = conv3x3(64, 32)
        self.dec_bn1b = nn.BatchNorm2d(32)
        self.dec_relu1b = nn.LeakyReLU(0.1)
        self.dec_conv1c = conv3x3(32, out_channels)
        self.dec_act = nn.Tanh()

    def forward(self, x):
        out_pool1 = self.pool1(self.enc_relu1(self.enc_bn1(self.enc_conv1(self.enc_relu0(self.enc_conv0(x))))))
        out_pool2 = self.pool2(self.enc_relu2(self.enc_bn2(self.enc_conv2(out_pool1))))
        out_pool3 = self.pool3(self.enc_relu3(self.enc_bn3(self.enc_conv3(out_pool2))))
        out_pool4 = self.pool4(self.enc_relu4(self.enc_bn4(self.enc_conv4(out_pool3))))
        out_pool5 = self.pool5(self.enc_relu5(self.enc_bn5(self.enc_conv5(out_pool4))))
        out = self.upsample5(self.enc_relu6(self.enc_bn6(self.enc_conv6(out_pool5))))
        out = self.upsample4(self.dec_relu5b(self.dec_bn5b(self.dec_conv5b(self.dec_relu5a(self.dec_bn5a(self.dec_conv5a(torch.cat((out, out_pool4), 1))))))))
        out = self.upsample3(self.dec_relu4b(self.dec_bn4b(self.dec_conv4b(self.dec_relu4a(self.dec_bn4a(self.dec_conv4a(torch.cat((out, out_pool3), 1))))))))
        out = self.upsample2(self.dec_relu3b(self.dec_bn3b(self.dec_conv3b(self.dec_relu3a(self.dec_bn3a(self.dec_conv3a(torch.cat((out, out_pool2), 1))))))))
        out = self.upsample1(self.dec_relu2b(self.dec_bn2b(self.dec_conv2b(self.dec_relu2a(self.dec_bn2a(self.dec_conv2a(torch.cat((out, out_pool1), 1))))))))
        out = self.dec_conv1c(self.dec_relu1b(self.dec_bn1b(self.dec_conv1b(self.dec_relu1a(self.dec_bn1a(self.dec_conv1a(torch.cat((out, x), 1))))))))
        out = self.dec_act(out)
        return out


    @property
    def model_size(self):
        return self._model_size()

    def _model_size(self):
        n_params, n_conv_layers = 0, 0
        for param in self.parameters():
            n_params += param.numel()
        for module in self.modules():
            if 'Conv' in module.__class__.__name__ \
                    or 'conv' in module.__class__.__name__:
                n_conv_layers += 1
        return n_params, n_conv_layers

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, 
        use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(image_channels, n_channels, 
            kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(n_channels, n_channels, 
            kernel_size=kernel_size, padding=padding, bias=False))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels, 
            kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

        print(f'model size: {self._model_size()}')

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        # out: residual, y: noisy input
        return y - out

    def _initialize_weights(self):
        print('init weight')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    @property
    def model_size(self):
        return self._model_size()
    
    def _model_size(self):
        n_params, n_conv_layers = 0, 0
        for param in self.parameters():
            n_params += param.numel()
        for module in self.modules():
            if 'Conv' in module.__class__.__name__ \
                    or 'conv' in module.__class__.__name__:
                n_conv_layers += 1
        return n_params, n_conv_layers
        
        
#training imports
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
import torch
import json
import random
import time
from pprint import pprint
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#load the model here
in_channels = 1
out_channels = 1
#device = '/CPU:0'
lr = 1e-4 #learning rate
wd = 1e-4 #weight decay
logger = {}
logger['rmse_train'] = []
logger['rmse_test'] = []
n_train_pixels = batch_size*im_height*im_width*4 #slices
n_test_pixels = batch_size_test*im_height*im_width*4 #here batch is only 1 sample

#model selction based on net value
# if net == 0: #unet batch norm
#     model = UnetN2N(in_channels, out_channels).to(device) #set the model and assign to the device too
# if net == 1: #dncnn
#     depth = 17
#     width = 64
#     model = DnCNN(depth=depth, n_channels=width, image_channels=1, use_bnorm=True, kernel_size=3).to(device) #DnCNN model

model = DenseED(in_channels=1, out_channels=1, 
                blocks=[8,8,8],
                growth_rate=16, 
                init_features=64,
                drop_rate=0.38,
                bn_size=8,
                bottleneck=False,
                out_activation=None).to(device)

# optim_parameters = {'lr': lr}
# optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd, betas=[0.9, 0.99], **optim_parameters)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3,
                       weight_decay=3e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8)
print('DnCNN Model summary >>>>>>>> ',model) #diasblaed model print
# print('DnCNN Model size >>>>>>> ',model.model_size)

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

train_len = np.floor(len(dataloader_train)/batch_size)
#print('train step size', train_len)
test_len = np.ceil(len(dataloader_test)/batch_size)
#print('test step size', test_len)
ckpt_dir = '/afs/crc.nd.edu/user/v/vmannam/Desktop/Summer20/May20/1905/Pytorch_training/saved_model/'
#train function
def generateNumber(num):
    mylist = []
    for i in range(num):
        mylist.append(i)
    return mylist


def train(epochs):
    iters = 0
    mse_loss_train = []
    mse_loss_test = []
    total_steps = epochs * len(dataloader_train)
    for epoch in range(1, epochs + 1):
        model.train()
        mse=0
        #mse_sample = 0
        for batch_idx, (noisy_input, clean_target) in enumerate(dataloader_train):
            iters += 1
            #print('Input shape here >>>>>>', noisy_input.shape)
            #noisy_input = torch.from_numpy(noisy_input)
            #clean_target = torch.from_numpy(clean_target)
            noisy_input = torch.squeeze(noisy_input, 0) #from 5D to 4D conversion
            clean_target = torch.squeeze(clean_target, 0) #from 5D to 4D conversion
            noisy_input = noisy_input.float()
            clean_target = clean_target.float()
            noisy_input, clean_target = noisy_input.to(device), \
                clean_target.to(device)
            
            model.zero_grad()
            denoised = model(noisy_input)
            
            #convert to 256x256 image
            full_size = 256 
            part_size = 128
            denoised_full = torch.zeros(batch_size, 1, full_size, full_size)
            clean_full = torch.zeros(batch_size, 1, full_size, full_size)
            for i in range(batch_size):
                #print('xxxxx >>>>', i)
                denoised_full[i, 0, 0:part_size, 0:part_size] = denoised[i*4,:,:,:]
                denoised_full[i, 0, 0:part_size, part_size:full_size] = denoised[(i*4)+1,:,:,:]
                denoised_full[i, 0, part_size:full_size, 0:part_size] = denoised[(i*4)+2,:,:,:]
                denoised_full[i, 0, part_size:full_size,part_size:full_size] = denoised[(i*4)+3,:,:,:]
                
                clean_full[i, 0, 0:part_size, 0:part_size] = clean_target[(i*4),:,:,:]
                clean_full[i, 0, 0:part_size, part_size:full_size] = clean_target[(i*4)+1,:,:,:]
                clean_full[i, 0, part_size:full_size, 0:part_size] = clean_target[(i*4)+2,:,:,:]
                clean_full[i, 0, part_size:full_size,part_size:full_size] = clean_target[(i*4)+3,:,:,:]
                
            loss = F.mse_loss(denoised_full, clean_full, size_average=False) #this is the error bwtween images of 256x256 MSE loss
            loss.backward()

            step = epoch * len(dataloader_train) + batch_idx + 1
            pct = step / total_steps
            #lr = scheduler.step(pct)
            #adjust_learning_rate(optimizer, lr)
            optimizer.step()

            mse += loss.item()
            #mse_sample = loss.item() #this is the present loss
        rmse = np.sqrt(mse / n_train_pixels)
        scheduler.step(rmse)
        
        logger['rmse_train'].append(rmse)
        print("Epoch {} training RMSE: {:.6f}".format(epoch, rmse))
            #sys.exit(0)
        
        mse_loss_train.append(rmse)
        # test ------------------------------
        with torch.no_grad():
            model.eval()
            mse_test = 0.
            for batch_idx, (noisy, clean) in enumerate(dataloader_test):
                #noisy = torch.from_numpy(noisy)
                #clean = torch.from_numpy(clean)
                noisy = torch.squeeze(noisy, 0) #from 5D to 4D conversion
                clean = torch.squeeze(clean, 0) #from 5D to 4D conversion
                noisy = noisy.float()
                clean = clean.float()
                noisy, clean = noisy.to(device), clean.to(device)
                denoised = model(noisy)
                loss = F.mse_loss(denoised, clean, size_average=False)
                mse_test += loss.item()
                par_babu = denoised.cpu().detach().numpy() 
                tar_babu = clean.cpu().detach().numpy() 
                if epoch % 50 == 0:
                    io.savemat('target_pressure_%d.mat'%epoch, dict([('target_pressure1_test',np.array(tar_babu))]))
                    io.savemat('predict_pressure_%d.mat'%epoch, dict([('predict_pressure1_test',np.array(par_babu))])) 

            rmse_test = np.sqrt(mse_test/ n_test_pixels)
            logger['rmse_test'].append(rmse_test)
            print("Epoch {}: test RMSE: {:.6f}".format(epoch, rmse_test))
        mse_loss_test.append(rmse_test)
    
    #end of for loop so saving results here    
    timestr_plt = time.strftime("%Y%m%d_%H%M%S")
    np.savetxt(path+'/results/'+'DnCNN_train_RMSE_1905_'+timestr_plt+'.txt', np.array(mse_loss_train))
    np.savetxt(path+'/results/'+'DnCNN_test_RMSE_1905_'+timestr_plt+'.txt', np.array(mse_loss_test))
    # save model
    torch.save(model.state_dict(), ckpt_dir + "/DnCNN_model_epoch{}.pth".format(epoch))
    torch.save(model,ckpt_dir+'DnCNN_model_SR_1905_'+timestr_plt+'.pt')
    #plots 
    xp = np.array(generateNumber(epochs))
    loss_train = np.array(mse_loss_train)
    loss_test = np.array(mse_loss_test)
    plt.plot(xp, loss_train, 'b*', label='Train_loss')
    plt.plot(xp, loss_test, 'rs', label='Test_loss')
    plt.xticks(np.arange(min(xp), max(xp)+1, 1.0))
    plt.xlabel('Epochs ')
    plt.ylabel('Train and Test Loss ')
    plt.title('Training SR dataset ')
    plt.legend(loc='best', frameon=False)
    
    plt.savefig(path+'results/'+'DnCNN_RMSE_loss_1905_'+timestr_plt+'.png')

if __name__ == "__main__":
    #Trainer(default_save_path=’/your/path/to/save/checkpoints’)
    path = '/afs/crc.nd.edu/user/v/vmannam/Desktop/Summer20/May20/1905/Pytorch_training/'
    #with open(path + "/args.txt", 'w') as args_file:
        #json.dump(logger, args_file, indent=4)
    Epochs =  200  
        #Trainer(default_save_path=path)
    print('Start training........................................................')
    t1 = time.time()
    train(Epochs)
    t2 = time.time()
    #torch.save(model_H,'ConvNet2311_Hadamard_03.pt')
    print('Execution time >>>>>>>:', t2-t1)