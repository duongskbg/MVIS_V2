import cv2, torch, os, copy, glob
from models.modelPath import mGetPath
import matplotlib.pyplot as plt

def predict_boxlist(model, img):
    results = model(img)
    results = results.pandas().xywh[0] # predictions
    results = results[results['confidence']>0.6]
    results = results.values.tolist()
    return results

def save_image_to_folder(img, imgNum, folderNum):
    img = cv2.resize(img, (300, 300))
    cv2.imwrite( 'Data/Folder' + str(folderNum) + '/img' + str(int(imgNum)) + '.jpg', img)

def draw_rects(img, results):
    for res in results:
        x, y, w, h = res[0], res[1], res[2], res[3]
        conf = round(res[4], 2)
        x1 = int(x-w/2)
        y1 = int(y-h/2)
        x2 = int(x+w/2)
        y2 = int(y+h/2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        img = cv2.putText(img, str(conf), (x1+100, y1), cv2.FONT_HERSHEY_SIMPLEX, 
                   2, (255, 0, 0), 2, cv2.LINE_AA)
    return img

""" crop from video """
model = torch.hub._load_local(mGetPath.dataPath, 'custom', mGetPath.modelPath)
cap = cv2.VideoCapture(mGetPath.videoPath)
imgNum = 0
while imgNum < 10000:   
    _, img = cap.read()
    if img is None:
        break
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preds = predict_boxlist(model, img)
    if len(preds) == 0:
        continue
    for i, pred in enumerate(preds):
        x, y, w, h = pred[0], pred[1], pred[2], pred[3]
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        # # draw to test the detection
        # img = draw_rects(img, [pred])
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # # --------------------------
        cropImg = img[y1:y2, x1:x2]
        img1 = cropImg[:int(h/2), :int(w/3)]
        img2 = cropImg[:int(h/2): , int(w/3):int(2*w/3)]
        img3 = cropImg[:int(h/2):, int(2*w/3):]
        img4 = cropImg[int(h/2):, int(3*w/4):]
        cv2.imwrite( 'Data/img' + str(int(imgNum)) + '.jpg', img)
        save_image_to_folder(img1, imgNum, 1)
        save_image_to_folder(img2, imgNum, 2)
        save_image_to_folder(img3, imgNum, 3)
        save_image_to_folder(img4, imgNum, 4)
        imgNum += 1
cap.release()
cv2.destroyAllWindows()

# """ crop from folder """
# model = torch.hub._load_local(mGetPath.dataPath, 'custom', mGetPath.modelPath)
# files = glob.glob( 'D:/Minh/Projects/MIVIS/Data/PickedImages/images/*.jpg' )
# imgNum = 0
# for file in files:
#     img = cv2.imread(file)
#     #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     preds = predict_boxlist(model, img)
#     if len(preds) == 0:
#         continue
#     for i, pred in enumerate(preds):
#         x, y, w, h = pred[0], pred[1], pred[2], pred[3]
#         x1 = int(x - w/2)
#         y1 = int(y - h/2)
#         x2 = int(x + w/2)
#         y2 = int(y + h/2)
#         # # draw to test the detection
#         # img = draw_rects(img, [pred])
#         # cv2.imshow('img', img)
#         # cv2.waitKey(0)
#         # # --------------------------
#         cropImg = img[y1:y2, x1:x2]
#         img1 = cropImg[:int(h/3), :int(w/4)]
#         img2 = cropImg[:int(h/3): , int(3*w/8):int(5*w/8)]
#         img3 = cropImg[:int(h/3):, int(3*w/4):]
#         img4 = cropImg[int(h/2):, int(3*w/4):]
#         #save_image_to_folder(img1, imgNum, 1)
#         #save_image_to_folder(img2, imgNum, 2)
#         save_image_to_folder(img3, imgNum, 3)
#         #save_image_to_folder(img4, imgNum, 4)
#         imgNum += 1