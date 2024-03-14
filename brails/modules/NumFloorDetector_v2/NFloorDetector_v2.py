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
#
# Last updated:
# 05-08-2022

import os
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
import warnings
import sys
sys.path.append("..")
from brails.modules.CLIPClassifier.CLIPClassifier import CLIPClassifier
#from CLIPClassifier import CLIPClassifier


warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    useGPU=True
else:    
    useGPU=False

'''
NFloorDetector with CLIP
'''
class NFloorDetector_v2():
    def __init__(self):
        
        self.default_text_prompts = [
            'one story house','bungalow','flat house', #one-story prompts
            'two story house','two-story duplex','raised ranch', #two-story prompts
            'three story house','three story house','three-decker' #three-story prompts
        ]
        self.default_classes = [1,2,3]
        self.model = CLIPClassifier(self.default_text_prompts)
        self.system_dict = {}
        self.system_dict["train"] = {}
        self.system_dict["train"]["data"] = {}
        self.system_dict["train"]["model"] = {}        
        self.system_dict["infer"] = {}
        self.system_dict["infer"]["params"] = {}

        self.set_fixed_params()
    
    def set_fixed_params(self):        
        self.system_dict["train"]["data"]["trainSet"] = "train"
        self.system_dict["train"]["data"]["validSet"] = "valid"
        self.system_dict["train"]["data"]["classes"] = ["floor"]
        self.system_dict["train"]["model"]["valInterval"] = 1
        self.system_dict["train"]["model"]["saveInterval"] = 5
        self.system_dict["train"]["model"]["esMinDelta"] = 0.0
        self.system_dict["train"]["model"]["esPatience"] = 0
        
    def load_train_data(self, rootDir="datasets/", nWorkers=0, batchSize=2):
        self.system_dict["train"]["data"]["rootDir"] = rootDir
        self.system_dict["train"]["data"]["nWorkers"] = nWorkers
        self.system_dict["train"]["data"]["batchSize"] = batchSize        
        
    '''
    def train(self, optim="adamw", lr=1e-4, numEpochs=25, nGPU=1):
        self.system_dict["train"]["model"]["topOnly"] = topOnly
        self.system_dict["train"]["model"]["optim"] = optim          
        self.system_dict["train"]["model"]["lr"] = lr
        self.system_dict["train"]["model"]["numEpochs"] = numEpochs
        self.system_dict["train"]["model"]["nGPU"] = nGPU        
        
 
        # Train    
        self.model.train(
            trainDataDir=self.system_dict["train"]["data"]["rootDir"], 
            batchSize=self.system_dict["train"]["data"]["batchSize"], 
            nepochs=numEpochs, plotLoss=True, text_prompts = self.default_text_prompts)
    '''

    '''  
    def retrain(self, optim="adamw", lr=1e-4, numEpochs=25, nGPU=1, model_path = None):
        self.system_dict["train"]["model"]["topOnly"] = False
        self.system_dict["train"]["model"]["optim"] = optim          
        self.system_dict["train"]["model"]["lr"] = lr
        self.system_dict["train"]["model"]["numEpochs"] = numEpochs
        self.system_dict["train"]["model"]["nGPU"] = nGPU        
                
        if(model_path==None):
            model_path = os.path.join('pretrained_weights',f"{self.model.model_arch}.pth")
        os.makedirs('pretrained_weights',exist_ok=True)
        if not os.path.isfile(model_path):
            print('Loading default floor detector model file to the pretrained folder...')
            torch.hub.download_url_to_file('https://zenodo.org/record/4421613/files/efficientdet-d4_trained.pth',
                                           model_path, progress=False)
            
        self.model.retrain(
            modelPath=model_path, trainDataDir=self.system_dict["train"]["data"]["rootDir"], 
            batchSize=self.system_dict["train"]["data"]["batchSize"], 
            nepochs=numEpochs, plotLoss=True, text_prompts = self.default_text_prompts)
    '''

    def predict(self, images, 
                modelPath='tmp/models/efficientdet-d4_nfloorDetector.pth',
                gpuEnabled=useGPU, classes = None, text_prompts = None):
        self.system_dict["infer"]["images"] = images
        self.system_dict["infer"]["modelPath"] = modelPath
        self.system_dict["infer"]["gpuEnabled"] = gpuEnabled
        self.system_dict["infer"]['predictions'] = []
        
        print('\nDetermining the number of floors for each building...')
        # Start Program Timer
        startTime = time.time()
        
        # Get the Image List
        try: 
            imgList = os.listdir(self.system_dict["infer"]["images"])
            for imgno in range(len(imgList)):
                imgList[imgno] = os.path.join(self.system_dict["infer"]["images"],imgList[imgno])
        except:
            imgList = self.system_dict["infer"]["images"]
        
        nImages = len(imgList)
            
        # Create and Define the Inference Model
        
        print("Performing floor detections...")
        #return tuples of (img_path, pred)
        predictions = self.model.predict(modelPath=modelPath, testDataDir=imgList,
                classes=self.default_classes, text_prompts = text_prompts)
        #preds = [pred for im,pred in predictions]
        preds = predictions
        
        self.system_dict["infer"]['predictions'] = predictions
        self.system_dict["infer"]["images"] = imgList
        
        # End Program Timer and Display Execution Time
        endTime = time.time()
        hours, rem = divmod(endTime-startTime, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\nTotal execution time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        return preds

# classifier = NFloorDetector_v2()
# data_dir = '/nfs/turbo/coe-stellayu/brianwang/testData/nFloors/merged_data/Ann Arbor, MI'
# import glob
# images = list(glob.glob(f'{data_dir}/*'))[:20]
# preds = classifier.predict(
#     modelPath = '/nfs/turbo/coe-stellayu/brianwang/clip/ckpt/ViT-B-32.pt', 
#     images = images, 
# )
# print(preds)
