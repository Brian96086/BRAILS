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


from brails.modules.ImageClassifier.ImageClassifier import ImageClassifier

import torch
import os
import random

class OccupancyClassifier(ImageClassifier):

    def __init__(self, modelPath=None): 
    
        if modelPath == None:
            os.makedirs('tmp/models',exist_ok=True)
            modelPath = 'tmp/models/OccupancyClassifier_v1.pth'
            if not os.path.isfile(modelPath):
                print('Loading default occupancy classifier model file to tmp/models folder...')
                torch.hub.download_url_to_file('https://zenodo.org/record/7272099/files/trained_model_occupancy_v1.pth',
                                               modelPath, progress=False)
                print('Default occupancy classifier model loaded')
            else: 
                print(f"Default occupancy classifier model at {modelPath} loaded")
        else:
            print(f'Inferences will be performed using the custom model at {modelPath}')
     
        self.modelPath = modelPath
        self.classes = ['Other','Residential']  
        
    def predict(self, dataDir):
        imageClassifier = ImageClassifier()
        imageClassifier.predict(self.modelPath,dataDir,self.classes)
        def hazclass(str):
            if hazclass=='Residential':
                out = random.choice(['RES1','RES3'])
            else:
                out = 'COM1'
            return(out)
        self.preds = [hazclass(pred) for pred in imageClassifier.preds]
        
                      
        
    def retrain(self, dataDir, batchSize=8, nepochs=100, plotLoss=True):
        imageClassifier = ImageClassifier()
        imageClassifier.retrain(self.modelPath,dataDir,batchSize,nepochs,plotLoss)

if __name__ == '__main__':
    pass