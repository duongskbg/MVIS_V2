import cv2, copy, os, time
import tkinter as tk
import numpy as np
from models.modelMain import Camera, SmallYolo
from models.modelPath import mGetPath
from models.modelMain import CNN
from models.modelStatus import boxStatus, BoxStatistics
from Views.viewMain import View
from datetime import datetime, timedelta
from collections import Counter
from tkinter import filedialog, messagebox
from models.modelRS232 import RS232

class Controller():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('JBD_AI Pham Hoang Minh')
        
        self.view = View(self, self.root)
        self.streamFlag = False
        self.useVidFlag = False
        if self.useVidFlag:
            self.cap = cv2.VideoCapture(mGetPath.videoPath)    
        else: # use cam
            self.webCam = Camera(usbPort = 0)
            self.webCam.set_config()
        self.model = SmallYolo(gridSize = 7, numClasses = 2, thres = 0.5)
        self.modelName = 'UDMP00SD'
        self.read_md_name() # to update the combobox
        self.load_submodel_list()
        self.aoi = [1300, 650, 1900, 1080] # [x1, y1, x2, y2] of the area of interest
        self.isOk = True; # status is OK or not, to update the status label on the right side
        self.announceTxt = None # to update the scrolled list
        self.statusList = []
        self.analyzeCounterNum = 0
        self.analyzeCounterThres = 5        
        self.currTime = time.time()
        self.boxStats = BoxStatistics()
        self.currDate = None
        self.rs232 = RS232()
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        # set event when close window
        self.root.mainloop()
    
    def load_submodel_list(self):
        self.subImageSize = (300, 300, 3)
        self.submodelList = []
        for i in range(1, 5, 1):
            clf = CNN( self.subImageSize, 1)     
            path = os.path.join(mGetPath.dataPath, self.modelName, 'submodel_' + str(i) + '.h5')
            clf.model.load_weights( path )
            self.submodelList.append(clf)
    
    def on_menu_open(self):
        path = filedialog.askopenfilename()
        self.useVidFlag = True
        self.cap = cv2.VideoCapture(path) 
    
    def on_btn_start(self, evt):
        self.streamFlag = True
        self.start_streaming()

    def on_btn_stop(self, evt):
        self.streamFlag = False

    """" helper function for when pressing the START button """
    def start_streaming(self):
        if self.useVidFlag:
            _ret, imgOrig = self.cap.read()
        else:                
            _ret, imgOrig = self.webCam.camera.read() # is BGR
        if _ret is True:
            img = copy.deepcopy(imgOrig)
            """ analyze image """
            self.analyzeCounterNum += 1
            preds = self.predict_boxlist(img) 
            # self.view.report_error('Length of preditions is ' + str(len(preds)) + '\n')
            if self.analyzeCounterNum > self.analyzeCounterThres:
                self.analyzeCounterNum = 0
                if len(preds) > 0 : 
                    # list of all inner rois of the detected rois (preds)
                    innerRois = self.analyze_rois(img, preds)                    
                    img = self.draw_boxes(img, preds) 
                    img = self.draw_inner_rois(img, innerRois)
                    self.update_issue_list(preds, innerRois)     
                    
            
            self.view.update_canvas(img)
        """ looping the start_streaming process """
        if self.streamFlag:
            self.root.after(0, self.start_streaming)  
        
    
    def draw_boxes(self, img, preds):  
        img2 = copy.deepcopy(img)  
        for i, pred in enumerate(preds):
            x, y, w, h = pred[0], pred[1], pred[2], pred[3]
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            className = self.convert_label_to_string(pred[5])
            color = (255, 0, 0)
            cv2.rectangle(img2, (x1, y1), (x2, y2), color, thickness = 2)
            img2 = cv2.putText(img2, className, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
        cv2.rectangle(img2, (self.aoi[0], self.aoi[1]), (self.aoi[2], self.aoi[3]), (255, 0, 0), 2)
        return img2
    
    def draw_inner_rois(self, img, innerRois):
        for rois in innerRois:
            for roi in rois:
                x1, y1, x2, y2 = roi['location']
                if roi['label'] == 0:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                cv2.circle(img, (int((x1+x2)/2), int((y1+y2)/2)), 15, color, thickness = 2)
        return img
    
    def convert_label_to_string(self, label):
        with open(mGetPath.classesPath) as file:
            classes = file.readlines()
        ret = classes[label]
        ret = ret.replace('\n', '')
        return ret

    def convert_labels_to_string(self, labels):
        return [ self.convert_label_to_string(lab) for lab in labels ]
    
    def predict_boxlist(self, img):
        # self.view.report_error('Start updating results\n')
        #results2 = self.test_model(img)
        results = self.model.model(img)
        # self.view.report_error('Done updating results\n')
        results = results.pandas().xywh[0] # predictions
        results = results[results['confidence']>0.6]
        results = results.values.tolist()
        return results
    
    def analyze_rois(self, img, preds):
        self.roisList = []
        # loop over all detected boxes, find the inner rois
        for boxIdx, pred in enumerate(preds):
            x, y, w, h = pred[0], pred[1], pred[2], pred[3]
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            cropImg = img[y1:y2, x1:x2]
            self.roisList.append( self.analyze_inner_rois(cropImg, x, y, w, h) )

        return self.roisList
    
    def analyze_inner_rois(self, cropImg, x, y, w, h):
        global label0, label1, label2, label3
        img0 = cropImg[:int(h/3), :int(w/4)]
        img1 = cropImg[:int(h/3): , int(3*w/8):int(5*w/8)]
        img2 = cropImg[:int(h/3):, int(3*w/4):]
        img3 = cropImg[int(h/2):, int(3*w/4):]
        # cv2.imshow('img', img2)
        # cv2.waitKey(0)
        label0 = self.submodelList[0].model.predict(np.expand_dims(cv2.resize(img0, (300, 300)), axis = 0))
        label1 = self.submodelList[1].model.predict(np.expand_dims(cv2.resize(img1, (300, 300)), axis = 0))
        label2 = self.submodelList[2].model.predict(np.expand_dims(cv2.resize(img2, (300, 300)), axis = 0))
        label3 = self.submodelList[3].model.predict(np.expand_dims(cv2.resize(img3, (300, 300)), axis = 0))
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x1 + img0.shape[1]), int(y1 + img0.shape[0])
        roi0 = {'label' : int(label0[0,0]), 'location': (x1, y1, x2, y2) }
        x1, y1 = int(x - img1.shape[1]/2), int(y - h/2)
        x2, y2 = int(x1 + img1.shape[1]), int(y1 + img1.shape[0])
        roi1 = {'label' : int(label1[0,0]), 'location': (x1, y1, x2, y2) }
        x1, y1 = int(x + w/2 - img2.shape[1]), int(y - h/2)
        x2, y2 = int(x1 + img2.shape[1]), int(y1 + img2.shape[0])
        roi2 = {'label' : int(label2[0,0]), 'location': (x1, y1, x2, y2) }
        x1, y1 = int(x + w/2 - img3.shape[1]), int(y + h/2 - img3.shape[0])
        x2, y2 = int(x1 + img3.shape[1]), int(y1 + img3.shape[0])
        roi3 = {'label' : int(label3[0]), 'location': (x1, y1, x2, y2) }
        return [roi0, roi1, roi2, roi3]
    
    # check if box(es) (with label) is/are in the area of interest, return the box index (-1 if no box in the AOI)
    def box_in_aoi(self, boxes):
        aoi = self.aoi
        ret = -1
        for iBox, box in enumerate(boxes):
            x, y, w, h = box[0], box[1], box[2], box[3]
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
            if x1 > aoi[0] and x2 < aoi[2] and y1 > aoi[1] and y2 < aoi[3]:
                return iBox
        return ret
    
    def update_issue_list(self, boxes, innerRois):
        iBox = self.box_in_aoi(boxes)
        if iBox != -1: # inside a box
            rois = innerRois[iBox]
            status = ''
            for roi in rois:
                status += str( roi['label'] )
            self.statusList.append(status)
        else: # outside a box
            if len(self.statusList) > 0 and self.statusList[-1] != 'out': # update when leaving the area of interest
                counter = Counter(self.statusList)
                counter.pop('out')
                now = datetime.now()
                currStatus = counter.most_common()[0][0]
                
                txt = now.strftime("%d/%m/%Y_%H:%M:%S ") + boxStatus[currStatus] + '\n'
                currTime = time.time()
                # save report to log when there is a day shift (from today to tomorrow)
                if self.currDate is not None:
                    if now.day != self.currDate.day:
                        self.create_report()
                    else: # remove this 'else' if want to store only once per day
                        self.create_report()
                if currTime - self.currTime > 5: # if the gap between 2 updates is not too small
                    self.boxStats.update_stats(currStatus)
                    self.view.update_stats(self.boxStats)
                    self.view.report_error(txt)
                    self.rs232.send_data(txt)
                    self.write_to_file(mGetPath.logPath, txt)
                    self.currTime = currTime
                    self.currDate = now
                if currStatus == '1111':
                    self.view.update_status_icon('NORMAL')
                else:
                    self.view.update_status_icon('ERROR')
                self.statusList = [] # reset the status list
            self.statusList.append('out')

    def cbb_selected(self, event):
        self.modelName = self.view.cbbModels.get()
        self.load_submodel_list()
        
    def read_md_name(self):
        pathModelList = os.path.join( mGetPath.dataPath, 'models.txt' )
        with open(pathModelList, "r+") as modelList:
            models = modelList.readlines()
            self.view.cbbModels['values'] = [m[:-1] for m in models]
            self.view.cbbModels.current(0) # set default value of combo box
            
    def write_to_file(self, filePath, content):
        if not os.path.exists(filePath):
            with open( filePath, 'w') as f:
                f.write( content )
        else:
            with open( filePath, 'a') as f:
                f.write( content )
        
    def create_report(self):
        content = self.currDate.strftime("%d/%m/%Y_%H:%M:%S\n")
        passNum = self.view.passNum['text']
        failNum = self.view.failNum['text']
        yieldNum = self.view.yieldNum['text']
        content += f'STATS:\nPass: {passNum}\nFail: {failNum}\nYield: {yieldNum}\nTHREE COMMON MISSING\n'
        for commonError in self.view.commonErrorList:
            content += commonError['text']
            content += '\n'
        content += '-----------------------\n'
        self.write_to_file( mGetPath.log2Path, content )
        
    def on_close(self):
        response = messagebox.askyesno('Exit', 'Are you sure you want to exit?')
        if response:
            self.rs232.close_port()
            self.root.destroy()