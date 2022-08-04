from models.modelPath import mGetPath
import os, cv2
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import torch.nn as nn

class SmallYolo():
    def __init__(self, gridSize, numClasses, thres):
        self.gridSize = gridSize
        self.numClasses = numClasses
        self.thres = thres   
        self.model = torch.hub._load_local(mGetPath.dataPath, 'custom', mGetPath.modelPath)
        
class Camera():
    def __init__(self, usbPort = 0):
        self.usbPort = usbPort
        self.camera = cv2.VideoCapture(self.usbPort)

    def isOpened(self):
        return self.camera.isOpened()

    def take_pic(self):
        _, img = self.camera.read()
        return img
    
    def get_config(self):
        def get_value(lines, name):
            for line in lines:
                tmp = line.split(' ')
                if tmp[0] == name:
                    return float(tmp[-1])
        with open(mGetPath.configPath, 'r') as file:
            lines = file.readlines()
        lines = [ line[:-1] for line in lines]
        self.resolution_width = get_value(lines, 'resolution_width')
        self.resolution_height = get_value(lines, 'resolution_height')
        self.exposure = get_value(lines, 'exposure')
        self.brightness = get_value(lines, 'brightness')
        self.contrast = get_value(lines, 'contrast')
        self.hue = get_value(lines, 'hue')
        self.saturation = get_value(lines, 'saturation')
        self.sharpness = get_value(lines, 'sharpness')
        self.gamma = get_value(lines, 'gamma')
        self.white_balance = get_value(lines, 'white_balance')
        self.gain = get_value(lines, 'gain')
        
    def set_config(self):
        self.get_config()
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution_height)
        # self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        # self.camera.set(cv2.CAP_PROP_EXPOSURE, 3)
        
        #self.camera.set(cv2.CAP_PROP_CONTRAST, self.contrast)
        #self.camera.set(cv2.CAP_PROP_SATURATION, self.saturation)
        #self.camera.set(cv2.CAP_PROP_SHARPNESS, self.sharpness)
        #self.camera.set(cv2.CAP_PROP_GAMMA, self.gamma)
        #self.camera.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, self.white_balance)
        #self.camera.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, self.white_balance)
        # self.camera.set(cv2.CAP_PROP_AUTO_WB, 0)
        #self.camera.set(cv2.CAP_PROP_GAIN, self.gain)        
        
class CNN():
    def __init__(self, in_shape, out_shape = 1):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.model = self.model_create()        
        
    def model_create(self):
        self.model = Sequential()
        self.model.add( Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform',
                         input_shape = self.in_shape) )
        self.model.add( MaxPooling2D((2,2)) )
        self.model.add( Conv2D(32, (3, 3), activation = 'relu', strides = 3 ))
        self.model.add( MaxPooling2D((2,2)) )
        self.model.add( Conv2D(32, (3, 3), activation = 'relu', strides = 2) )
        self.model.add( MaxPooling2D((2,2)) )
        self.model.add(Flatten())
        self.model.add(Dense(100, activation = 'relu', kernel_initializer = 'he_uniform'))
        self.model.add(Dense(self.out_shape, activation = 'sigmoid'))
        self.model_compile()
        return self.model
    
    def model_compile(self):
        opt = Adam(learning_rate = 0.0001)
        self.model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
    def model_fit(self, X, y, epochs = 1000, batch_size = 5, validation_data = None, verbose = 1, class_weight = None):
        self.model.fit(X, y, epochs = epochs, batch_size = batch_size, validation_data = validation_data, 
                       class_weight=class_weight, verbose = verbose)
    
    def model_predict(self, X):
        return self.model.predict(X)
    
    def model_save(self, modelName):        
        self.model.save_weights(modelName)
    
"""" list of modules """
class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()
    def forward(self, x, augment = False, profile = False, visualize = False):
        y = []
        for module in self:
            y.append( module(x, augment, profile, visualize)[0] )
        y = torch.cat(y, 1)
        return y, None # inference, train output