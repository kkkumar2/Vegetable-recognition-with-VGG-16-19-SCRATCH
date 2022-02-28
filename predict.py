#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os

class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
        path_dir = 'D:\Data science\ineuron\Assignments\Vegetable vgg code'
        file_name = 'vgg16_transferlearning.h5'
        filepath = os.path.join(path_dir, file_name)
        model = load_model(filepath)

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        y_pred = np.argmax(result,axis=1)
        output_labels = ['Bean','Brinjal','Broccoli','Cabbage','Capsicum','Carrot','Cauliflower','Cucumber','Papaya','Potato','Radish','Tomato']
        return output_labels[y_pred[0]]


