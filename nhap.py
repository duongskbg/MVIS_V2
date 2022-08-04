import cv2, copy, glob
import numpy as np
from math import sqrt, atan2, pi
from scipy import ndimage
from models.modelMain import SmallYolo
import matplotlib.pyplot as plt

def get_lines(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize=3) 
    # Apply HoughLinesP method to directly obtain line end points
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=100, # Min number of votes for valid line
                minLineLength=20, # Min allowed length of line
                maxLineGap=10 # Max allowed gap between line for joining them
                )
    if lines is not None:
        lines = np.reshape(lines, (lines.shape[0], lines.shape[-1]))
    return lines

def convert_lines_to_polar(lines):
    ret = []
    for line in lines:
        x1, y1, x2, y2 = line
        radius = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
        theta = atan2( (y2 - y1), (x2 - x1) ) * 180 / pi
        ret.append( [radius, theta] )
    return np.array( ret )

def draw_boxes(img, preds):  
    img2 = copy.deepcopy(img)  
    for i, pred in enumerate(preds):
        x, y, w, h = pred[0], pred[1], pred[2], pred[3]
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        color = (255, 0, 0)
        cv2.rectangle(img2, (x1, y1), (x2, y2), color, thickness = 2)
    return img2

# rotate image according to the most detected lines
def rotate_image(img):
    lines = get_lines(img)
    if lines is None:
        return img
    polarLines = convert_lines_to_polar(lines)
    data = polarLines[:, 1]
    hList, xList, _ = plt.hist(data, bins = 30)
    maxIndex = np.argmax(hList)
    angle = xList[ maxIndex ]
    # choose the second big angle if the angle of rotation is too much
    while abs(angle) > 30:
        hList = np.delete( hList, maxIndex )
        xList = np.delete( xList, maxIndex )
        if len(hList) == 0:
            return img
        maxIndex = np.argmax(hList)
        angle = xList[ maxIndex ]
    rotated = ndimage.rotate(img, angle)
    return rotated

def crop_rotated_box(img, model, imgNum, fileName):  
    preds = model.model(img)
    preds = preds.pandas().xywh[0]
    preds = preds[preds['confidence']>0.6]
    preds = preds.values.tolist()
    for i, pred in enumerate(preds):
        x, y, w, h = pred[0], pred[1], pred[2], pred[3]
        x1 = int(x - w) if (x - w) > 0 else 0 
        y1 = int(y - h) if (y - h) > 0 else 0
        x2 = int(x + w) if (x + w) < img.shape[1] else img.shape[1]
        y2 = int(y + h) if (y + h) < img.shape[0] else img.shape[0]
        cropImg = img[y1 : y2, x1 : x2]
        rotatedImg = rotate_image(cropImg)
        #cv2.imwrite('Data/cropped' + str(imgNum) + '.jpg', cropImg)
        cv2.imwrite('Data/rotated' + str(imgNum) + '.jpg', rotatedImg)
        imgNum += 1
    return imgNum

#Read image
imgNum = 0
fileList = glob.glob('D:/Minh/Projects/MVISv2/Data/PickedImages/images/*.jpg')
myModel = SmallYolo(gridSize = 7, numClasses = 2, thres = 0.5)
for file in fileList:    
    image = cv2.imread(file)    
    imgNum = crop_rotated_box(image, myModel, imgNum, fileName=file)
    
# cropImg = cv2.imread('Data/cropped1.jpg')
# rotatedImg = rotate_image(cropImg)
# cv2.imshow('nbbsdg', rotatedImg)
# cv2.waitKey(0)

# lines = get_lines(img)
# polarLines = convert_lines_to_polar(lines)
# data = polarLines[:, 1]
# xyList, hList, wList = [], [], []
# for h in sns.histplot(data, bins = 20).patches:
#     xyList.append( h.get_xy() )
#     hList.append( h.get_height() )
#     wList.append( h.get_width() )
# maxIndex = np.argmax(hList)
# angle = xyList[ maxIndex ][0] + wList[ maxIndex ] / 2
# rotated = ndimage.rotate(img, angle)
# cv2.imshow('nbbsdg', rotated)
# cv2.waitKey(0)

# for line in lines:
#     img = cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0,255,0), thickness = 3)
# cv2.imshow('aaa', img)
# cv2.waitKey(0)