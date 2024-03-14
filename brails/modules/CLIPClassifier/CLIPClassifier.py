# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 The Regents of the University of California
#
# This file is part of BRAILS.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import glob
import copy
from PIL import Image
import sys
import requests
import zipfile
from clip.clip import load, tokenize
from clip.model import build_model
from clip.utils import preprocess_batch_img, predict_wrapper, pred_idx_to_labels
#import utils
class CLIPClassifier:

    def __init__(self, default_text_prompts, modelArch="ViT-B/32"): 
        #initializr model, predict, retrain() -> don't support train() (from scratch) for now
        self.modelArch = modelArch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.batchSize = None
        self.nepochs = None
        self.trainDataDir = None
        self.testDataDir = None
        self.classes = None
        self.lossHistory = None
        self.preds = None 
        self.default_text_prompts = default_text_prompts
        self.text_prompts = None #none unless set during training or inference

    def train(self, trainDataDir='tmp/hymenoptera_data', batchSize=8, nepochs=100, plotLoss=True, text_prompts = None):
        
        if trainDataDir=='tmp/hymenoptera_data':
            print('Downloading default dataset...')
            url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
            req = requests.get(url)
            zipdir = os.path.join('tmp',url.split('/')[-1])
            os.makedirs('tmp',exist_ok=True)
            with open(zipdir,'wb') as output_file:
                output_file.write(req.content)
            print('Download complete.')
            with zipfile.ZipFile(zipdir, 'r') as zip_ref:
                zip_ref.extractall('tmp')
        
        def train_model(model, tokenizer, dataloaders, criterion, optimizer, num_epochs=100, es_tolerance=10, text_prompts = None):
            since = time.time()
        
            val_acc_history = []
            
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0
            es_counter = 0
        
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
        
                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode
        
                    running_loss = 0.0
                    running_corrects = 0
        
                    # Iterate over data.
                    for image_inputs, labels in dataloaders[phase]:
                        image_inputs = image_inputs.to(self.device)
                        labels = labels.to(self.device)
                        self.text_prompts = text_prompts.to(self.device)
        
                        # zero the parameter gradients
                        optimizer.zero_grad()
        
                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training it has an auxiliary output. In train
                            #   mode we calculate the loss by summing the final output and the auxiliary output
                            #   but in testing we only consider the final output.
                            outputs = model(image_inputs, self.text_prompts)
                            loss = criterion(outputs, labels)
        
                            _, preds = torch.max(outputs, 1)
        
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
        
                        # statistics
                        running_loss += loss.item() * image_inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
        
                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase.capitalize(), epoch_loss, epoch_acc))
        
                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        es_counter = 0
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if phase == 'val':
                        es_counter += 1
                        val_acc_history.append(epoch_acc)
                if es_counter>=es_tolerance:
                  print('Early termination criterion satisfied.')
                  break
                print()
        
            time_elapsed = time.time() - since
            print('Best val Acc: {:4f}'.format(best_acc))
            print('Elapsed time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            # load best model weights
            model.load_state_dict(best_model_wts)
            return model, val_acc_history
        
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False
            else:
                for param in model.parameters():
                    param.requires_grad = True

        def initialize_model(model_name, use_pretrained=True):
            model_ft = load(model_name, self.device, 'tmp/models')
            return model_ft              
        
        self.batchSize = batchSize
        self.trainDataDir = trainDataDir

        classes = os.listdir(os.path.join(self.trainDataDir,'train'))        
        self.classes = sorted(classes)
        num_classes = len(self.classes)
        
        if isinstance(nepochs, int):
            nepochs_it = round(nepochs/2)
            nepochs_ft = nepochs - nepochs_it
        elif isinstance(nepochs, list) and len(nepochs)>=2:
            nepochs_it = nepochs[0]
            nepochs_ft = nepochs[1]
        else:
            sys.exit('Incorrect nepochs entry. Number of epochs should be defined as an integer or a list of two integers!')
            
        self.nepochs = [nepochs_it,nepochs_ft]
        
        # Initialize the model and associated data transform for this run
        model_ft, data_transforms = initialize_model(self.modelArch, num_classes, feature_extract=False, use_pretrained=True)
        model.train()
        
        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.trainDataDir, x), data_transforms) for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchSize, shuffle=True, num_workers=0) for x in ['train', 'val']}
        
        # Send the model to GPU
        model_ft = model_ft.to(self.device)
        
        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are 
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        
        # Observe that all parameters are being optimized
        optimizer_ft = optim.AdamW(params_to_update, lr=0.001, momentum=0.9)  
        
        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()
        
        # Train and evaluate
        model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=nepochs_it)
        print('New classifier head trained using transfer learning.')  
        
        # Initialize the non-pretrained version of the model used for this run
        print('\nFine-tuning the model...')
        set_parameter_requires_grad(model_ft,feature_extracting=False)
        final_model = model_ft.to(self.device)
        final_optimizer = optim.SGD(final_model.parameters(), lr=0.0001, momentum=0.9)
        final_criterion = nn.CrossEntropyLoss()
        _,final_hist = train_model(final_model, dataloaders_dict, final_criterion, final_optimizer, num_epochs=nepochs_ft)
        print('Training complete.')
        os.makedirs('tmp/models', exist_ok=True)
        torch.save(final_model.state_dict(), 'tmp/models/trained_model.pth')
        self.modelPath = 'tmp/models/trained_model.pth'
        
        
        # Plot the training curves of validation accuracy vs. number 
        #  of training epochs for the transfer learning method and
        #  the model trained from scratch
      
        plothist = [h.cpu().numpy() for h in hist] + [h.cpu().numpy() for h in final_hist]
        self.lossHistory = plothist
        if plotLoss:      
            plt.title("Validation Accuracy vs. Number of Training Epochs")
            plt.xlabel("Training Epochs")
            plt.ylabel("Validation Accuracy")
            plt.plot(range(1,len(plothist)+1),plothist)
            plt.ylim((0.4,1.))
            plt.xticks(np.arange(1, len(plothist)+1, 1.0))
            plt.show()

    def retrain(self, modelPath='tmp/models/trained_model.pth', 
                trainDataDir='tmp/hymenoptera_data', batchSize=8, 
                nepochs=100, plotLoss=True, text_prompts = None):
        
        if trainDataDir=='tmp/hymenoptera_data':
            print('Downloading default dataset...')
            url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
            req = requests.get(url)
            zipdir = os.path.join('tmp',url.split('/')[-1])
            os.makedirs('tmp',exist_ok=True)
            with open(zipdir,'wb') as output_file:
                output_file.write(req.content)
            print('Download complete.')
            with zipfile.ZipFile(zipdir, 'r') as zip_ref:
                zip_ref.extractall('tmp')
        
        def train_model(model, dataloaders, criterion, optimizer, num_epochs=100, es_tolerance=10, text_prompts = None):
            self.prompts = text_prompts if text_prompts!=None else self.default_text_prompts
            text_input = torch.cat([tokenize("a photo of a {}".format(c)) for c in self.prompts]).to(self.device)
            since = time.time()
        
            val_acc_history = []
            
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0
            es_counter = 0
        
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
        
                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode
        
                    running_loss = 0.0
                    running_corrects = 0
        
                    # Iterate over data.
                    for image_inputs, labels in dataloaders[phase]:
                        image_inputs = image_inputs.to(self.device)
                        labels = labels.to(self.device)
        
                        # zero the parameter gradients
                        optimizer.zero_grad()
        
                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training it has an auxiliary output. In train
                            #   mode we calculate the loss by summing the final output and the auxiliary output
                            #   but in testing we only consider the final output.
                            outputs = model(image_inputs, text_input)
                            loss = criterion(outputs, labels)
        
                            _, preds = torch.max(outputs, 1)
        
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
        
                        # statistics
                        running_loss += loss.item() * image_inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
        
                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase.capitalize(), epoch_loss, epoch_acc))
        
                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        es_counter = 0
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if phase == 'val':
                        es_counter += 1
                        val_acc_history.append(epoch_acc)
                if es_counter>=es_tolerance:
                  print('Early termination criterion satisfied.')
                  break
                print()
        
            time_elapsed = time.time() - since
            print('Best val Acc: {:4f}'.format(best_acc))
            print('Elapsed time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            # load best model weights
            model.load_state_dict(best_model_wts)
            return model, val_acc_history
        
        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False
            else:
                for param in model.parameters():
                    param.requires_grad = True

        
        self.batchSize = batchSize
        self.trainDataDir = trainDataDir

        classes = os.listdir(os.path.join(self.trainDataDir,'train'))        
        self.classes = sorted(classes)
        
        if isinstance(nepochs, int):
            self.nepochs = [0, nepochs]
        else:
            sys.exit('Incorrect nepochs entry. For retraining, number of epochs should be defined as an integer')

        
        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.trainDataDir, x), data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchSize, shuffle=True, num_workers=0) for x in ['train', 'val']}
        
        # Send the model to GPU
        if(os.path.isfile(model_path)):
            state_dict = load(modelPath, self.device)  #load given model checkpoint
        else:
            print(f'{modelPath} not found, loading a newly pretrained model')
            if(os.path.exist('tmp/models')): os.makedirs('tmp/models',exist_ok = True)
            state_dict = load(self.model_arch, self.device, 'tmp/models')
        model, data_transforms = build_model(state_dict) 
        model.train()

        set_parameter_requires_grad(model,feature_extracting=False)
        final_model = model.to(self.device)
        final_optimizer = optim.AdamW(final_model.parameters(), lr=0.0001, momentum=0.9)
        final_criterion = nn.CrossEntropyLoss()
        print(f'\nRetraining the model using the data located in {self.trainDataDir} folder...')
        _,final_hist = train_model(final_model, dataloaders_dict, final_criterion, final_optimizer, num_epochs=nepochs)
        print('Training complete.')
        os.makedirs('tmp/models', exist_ok=True)
        torch.save(final_model.state_dict(), 'tmp/models/retrained_model.pth')
        self.modelPath = 'tmp/models/retrained_model.pth'
        
        
        # Plot the training curves of validation accuracy vs. number 
        #  of training epochs for the transfer learning method and
        #  the model trained from scratch
        plothist = [h.cpu().numpy() for h in final_hist]
        self.lossHistory = plothist
        
        if plotLoss:        
            plt.plot(range(1,len(plothist)+1),plothist)
            plt.title("Validation Accuracy vs. Number of Training Epochs")
            plt.xlabel("Training Epochs")
            plt.ylabel("Validation Accuracy")
            #plt.ylim((0.4,1.))
            plt.xticks(np.arange(1, len(plothist)+1, 1.0))
            plt.show()    

    def predict(self, modelPath='tmp/models/trained_model.pth', 
                testDataDir='tmp/hymenoptera_data/val/ants',
                classes=['Ants','Bees'], text_prompts = None):
        self.modelPath = modelPath
        self.testDataDir = testDataDir
        self.classes = sorted(classes)
        self.prompts = text_prompts if text_prompts!=None else self.default_text_prompts
        text_input = torch.cat([tokenize("a photo of a {}".format(c)) for c in self.prompts]).to(self.device)
        
        def isImage(im):
            return im.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        
        model, data_transforms = load(modelPath, self.device, modelPath)
        model.eval()
        
        preds = []
        if isinstance(testDataDir,list):
            imlist = testDataDir[:]
            imlist = [im for im in imlist if isImage(im)]
            imlist.sort()
            pred_df = predict_wrapper(
                model, text_input, imlist, data_transforms, self.device, 
                batch_size = 200, agg = "max", num_classes = len(self.classes)
            )
            pred_df = pred_idx_to_labels(self.classes, pred_df)
            preds = [(im, pred) for im, pred in zip(imlist, pred_df['predictions'].tolist())]
            self.preds = preds 
        elif os.path.isdir(testDataDir):
            #imlist = os.listdir(testDataDir)
            imlist = glob.glob(f'{testDataDir}/*')
            imlist = [im for im in imlist if isImage(im)]
            imlist.sort()
            pred_df = predict_wrapper(
                model, text_input, imlist, data_transforms, self.device, 
                batch_size = 200, agg = "max", num_classes = len(self.classes)
            )
            pred_df = pred_idx_to_labels(self.classes, pred_df)
            #get filename by truncating the testDataDir
            preds = [(im[len(testDataDir)+1:], pred) for im,pred in zip(imlist, pred_df['predictions'].tolist())]
            self.preds = preds                  
        elif os.path.isfile(testDataDir) and isImage(testDataDir):
            img = plt.imread(testDataDir)[:,:,:3]
            pred_df = predict_wrapper(
                model, text_input, imlist, data_transforms, self.device, 
                batch_size = 1, agg = "max", num_classes = len(self.classes)
            )
            pred_df = pred_idx_to_labels(self.classes, pred_df)
            pred = pred_df['predictions']
            plt.imshow(img)
            plt.title((f"Predicted class: {pred}"))
            plt.show()
            print((f"Predicted class: {pred}"))
            self.preds = pred
        return self.preds

# if __name__ == '__main__':
#     pass

# classifier = CLIPClassifier(default_text_prompts = ['one story house', 'two story house', 'three story house'])
# classifier.predict(
#     modelPath = '/nfs/turbo/coe-stellayu/brianwang/clip/ckpt/ViT-B-32.pt', 
#     testDataDir = '/nfs/turbo/coe-stellayu/brianwang/testData/nFloors/merged_data/Ann Arbor, MI', 
#     classes = [1,2,3]
# )