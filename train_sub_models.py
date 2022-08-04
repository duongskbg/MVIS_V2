import os, glob, cv2
from models.modelPath import mGetPath
from models.modelMain import CNN
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.utils import class_weight

def feature_extraction(folderNum):   
    X, y = [], []
    labels = [0, 1]
    for lab in labels:
        imgPath = os.path.join(mGetPath.dataPath, 'Folder' + str(folderNum), str(lab))
        images = glob.glob( os.path.join(imgPath, '*.jpg') )            
        for im in images:
            im = cv2.imread(im)
            #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            X.append(im)
            y.append(lab)
    X, y = np.array(X), np.array(y)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

def classifier_extraction():
    #for folderNum in range(1, 5, 1):
    for folderNum in [3]:
        X, y = feature_extraction(folderNum)
        clf = CNN( X.shape[1:], 1)
        class_weights = None
        # class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
        # class_weights = dict(enumerate(class_weights))
        clf.model_fit(X, y, epochs = 300, class_weight = class_weights)
        #modelName = 'Data/submodel_' + str(folderNum) + '.h5'
        modelName = 'Data/submodel_' + str(folderNum) + '.h5'
        clf.model_save(modelName)
        print(f'Saved model {modelName}')
        clf = None

classifier_extraction()

# # test on an image
# model = CNN((300,300,3), 1)
# model.model.load_weights('Data/submodel_3.h5')
# for i in range(0,19,1):
#     #x = cv2.imread('Data/Folder3/1/img' + str(i) + '.jpg')
#     x = cv2.imread('Data/Folder3/0/img' + str(i) + '.jpg')
#     #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#     aa = model.model.predict(np.expand_dims(x, axis = 0))
#     print(f'Image number {i} has probablity {aa[0, 0]}')