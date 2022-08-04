# """" crop images from video"""
# import cv2
# cam = cv2.VideoCapture(r'Data/UDMP00SD.avi')
# imgNum = 0
# while(True):
#     _, img = cam.read()    
#     if img is None:
#         break
#     #img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
#     #img = img[int(0.6 * img.shape[0]):, :]
#     cv2.imwrite( 'Data/AllImages/img' + str(int(imgNum)) + '.jpg', img)
#     imgNum += 1 
# cam.release()
# cv2.destroyAllWindows()

# """ random pick photos to train """
# import glob, shutil, os
# from random import randint
# def get_filename_from_path(path):
#     loc = path.index('\\')
#     return path[ (loc + 1): ]
# def get_filenames_from_path(path, extension):
#     ret = []
#     for file in glob.glob( os.path.join(path, extension) ):
#         ret.append(get_filename_from_path(file))
#     return ret
# source = 'D:/Minh/Projects/MIVIS/Data/AllImages'
# destination = 'D:/Minh/Projects/MIVIS/Data/PickedImages'
# # checkPath = 'D:/Minh/Projects/yolov5/data/images/train'
# numPhotos = 1000
# listFiles = glob.glob( os.path.join(source, '*.jpg') )
# # filesListToCheck = get_filenames_from_path(checkPath, extension = '*.jpg')
# while numPhotos > 0:    
#     i = randint(0, len(listFiles))
#     # if get_filename_from_path( listFiles[i] ) in filesListToCheck:
#     #     continue
#     shutil.copy(listFiles[i], destination)
#     numPhotos -= 1
    
""" remove redundant images (as some images do not have any labels) """
import os, glob
def get_filename_from_path(path): # without extension
    loc = path.index('\\')
    return path[ (loc + 1): (-4)]
def get_filenames_from_path(path, extension): # without extension
    ret = []
    for file in glob.glob( os.path.join(path, extension) ):
        ret.append(get_filename_from_path(file))
    return ret
listFiles1 = glob.glob('D:/Minh/Projects/MIVIS/Data/PickedImages/images/*.jpg')
listFiles2 = get_filenames_from_path( path = 'D:/Minh/Projects/MIVIS/Data/PickedImages/labels', extension = '*.txt')
for file1 in listFiles1:
    if get_filename_from_path(file1) not in listFiles2:
        os.remove( file1 )