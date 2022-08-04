from tkinter import Frame, Canvas, Button, Label, LabelFrame, PhotoImage, Menu
from tkinter.ttk import Combobox
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import ttk, scrolledtext

class View():
    def __init__(self, controller, root):
        self.root = root
        self.root.iconphoto(False, PhotoImage(file = 'data/app_icon.png'))
        self.controller = controller
        
        self.startImage = tk.PhotoImage( file = 'data/play.png')
        self.pauseImage = tk.PhotoImage( file = 'data/pause.png')
        
        self.create_menu()
        self.create_top_bar()
        self.create_drawing_canvas()
        self.create_right_section()        
        
    def create_top_bar(self):
        self.topBar = LabelFrame(self.root)
        self.topBar.grid(row = 0, column = 0, columnspan = 2, sticky = 'WE')
        self.title = Label(self.topBar, text = 'MVIS', width = 95, font = ('Courier', 15, 'bold'), fg = 'white', bg = 'dodgerblue')
        self.title.grid(row = 0, column = 0, sticky = 'WE')

        self.modelLabel = Label(self.topBar, text='Model', width=6, background='white')
        self.modelLabel.grid(row = 0, column = 1, sticky = 'E')
        self.cbbModels = Combobox(self.topBar, width=8)
        self.cbbModels['state'] = 'readonly'
        self.cbbModels.grid(row = 0, column = 2, sticky = 'E') 
        self.cbbModels.bind('<<ComboboxSelected>>', self.controller.cbb_selected)
        
        for child in self.topBar.winfo_children():
            child.grid_configure(padx = 5, pady = 5, ipadx = 5, ipady = 5)
    
    def create_menu(self):
        # Menu
        self.menuBar = Menu(self.root)
        # FILE layout
        self.fileMenu = Menu(self.menuBar, tearoff=0)         
        # add options to "File"
        self.fileMenu.add_command(label="Open...", command=self.controller.on_menu_open)
        self.fileMenu.add_separator()        
        self.fileMenu.add_command(label="Exit", command=self.root.destroy)
        self.menuBar.add_cascade(label="File", menu=self.fileMenu)
        self.root.config(menu=self.menuBar)    
    
    def create_drawing_canvas(self):
        self.imgCanvas = Canvas(self.root, width = 800, height = 600)
        self.imgCanvas.grid(row = 1, column = 0, sticky = 'NSWE')

    def create_right_section(self):
        self.rightSection = LabelFrame(self.root)
        self.rightSection.grid(row = 1, column = 1, rowspan = 2, sticky = 'ESNW')        
        self.create_button_bar()  
        self.create_right_mid_bar()        
        """" Issue lists """
        self.reportFrame = LabelFrame(self.rightSection)
        self.reportFrame.grid(row = 2, column = 0, sticky = 'WE')
        Label(self.reportFrame, text = 'Issue Lists', font = ('Courier', 12)).grid(row = 0, column = 0)
        self.reportTxt = scrolledtext.ScrolledText(self.reportFrame, width = 40, height = 18, wrap = tk.WORD)
        self.reportTxt.grid(row = 1, column = 0, padx = 5, pady = 5)  
        """ Statistic box """
        self.statFrame = LabelFrame(self.rightSection)
        self.statFrame.grid(row = 3, column = 0, sticky = 'WE')
        fontSize = 10
        Label(self.statFrame, text = 'STATS', font = ('Courier', fontSize)).grid(row = 0, column = 0)
        Label(self.statFrame, text = 'Pass: ', font = ('Courier', fontSize)).grid(row = 1, column = 0)
        Label(self.statFrame, text = 'Fail: ', font = ('Courier', fontSize)).grid(row = 2, column = 0)
        Label(self.statFrame, text = 'Yield: ', font = ('Courier', fontSize)).grid(row = 3, column = 0)
        self.passNum = Label(self.statFrame, text = '', font = ('Courier', fontSize))
        self.passNum.grid(row = 1, column = 1)
        self.failNum = Label(self.statFrame, text = '', font = ('Courier', fontSize))
        self.failNum.grid(row = 2, column = 1)
        self.yieldNum = Label(self.statFrame, text = '', font = ('Courier', fontSize))
        self.yieldNum.grid(row = 3, column = 1)
        Label(self.statFrame, text = 'THREE COMMON MISSING', font = ('Courier', fontSize )).grid(row = 0, column = 2)
        self.commonErrorList = [
            Label(self.statFrame, text = '', font = ('Courier', fontSize)),
            Label(self.statFrame, text = '', font = ('Courier', fontSize)),
            Label(self.statFrame, text = '', font = ('Courier', fontSize))
            ]
        for i, commonError in enumerate(self.commonErrorList):
            commonError.grid(row = i + 1, column = 2, sticky = 'E')
        
        for child in self.rightSection.winfo_children():
            child.grid_configure(padx = 5, pady = 5, ipadx = 5, ipady = 5)
        
    def create_button_bar(self):
        self.buttonBar = LabelFrame(self.rightSection)
        self.buttonBar.grid(row = 0, column = 0, sticky = 'WE')        
        self.buttonStart = Button(self.buttonBar, image = self.startImage)
        self.buttonStart.bind('<Button-1>', self.controller.on_btn_start)
        self.buttonStart.grid(row = 0, column = 0)        
        self.buttonPause = Button(self.buttonBar, image = self.pauseImage)
        self.buttonPause.bind('<Button-1>', self.controller.on_btn_stop)
        self.buttonPause.grid(row = 0, column = 1)
        for child in self.buttonBar.winfo_children():
            child.config(height = 70, width = 100)
            child.grid_configure(padx = 5, pady = 5, ipadx = 5, ipady = 5)        

    def create_right_mid_bar(self):
        self.rightMidBar = LabelFrame(self.rightSection)
        self.rightMidBar.grid(row = 1, column = 0, sticky = 'WE')        
        self.statusIcon = Label(self.rightMidBar, text = 'Normal', width = 26, height = 4, font = ('Courier', 16, 'bold'))
        self.statusIcon.config(fg = 'white', bg = 'limegreen')
        self.statusIcon.grid(row = 0, column = 0)
        self.statusIcon.grid_configure(padx = 5, pady = 5)
        
    def convert_image_to_display(self, img):
        img = cv2.resize(img, (800, 600))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pilImg = Image.fromarray(img)
        tkImg = ImageTk.PhotoImage(pilImg)
        return tkImg

    def update_canvas(self, img):
        global tkImg
        tkImg = self.convert_image_to_display(img)
        self.imgCanvas.create_image(0, 0, anchor = 'nw', image = tkImg)
        self.imgCanvas.update()
        
    def update_status_icon(self, txt):
        if txt == 'NORMAL':
            self.statusIcon.config(text = 'NORMAL', fg = 'white', bg = 'limegreen')
        else:
            self.statusIcon.config(text = 'ERROR', fg = 'white', bg = 'crimson') #'crimson' 
            
    def report_error(self, txt):
        self.reportTxt.insert( tk.END, txt )
    
    # return a list of 'numError' components, which is the list of most common errors and their #occurences
    def most_common_error(self, boxStats, numError = 3):
        error_keys = ['power_cord_fail', 'left_ear_fail', 'right_ear_fail', 'booklet_fail']
        error_dict = { one_key : boxStats.count_dict[one_key] for one_key in error_keys }
        keys = list( error_dict.keys() )
        values = list( error_dict.values() )
        ret = []
        for i in range(numError):
            idx = values.index(max(values))
            if values[idx] == 0:
                continue
            if keys[idx] == 'power_cord_fail':
                txt = 'Power cord'
            elif keys[idx] == 'left_ear_fail':
                txt = 'Left ear'
            elif keys[idx] == 'right_ear_fail':
                txt = 'Right ear'
            elif keys[idx] == 'booklet_fail':
                txt = 'Booklet'
            ret.append((txt, values[idx]))
            keys.pop(idx)
            values.pop(idx)
        return ret
        
    def update_stats(self, boxStats):
        self.passNum['text'] = boxStats.pass_time
        self.failNum['text'] = boxStats.fail_time
        self.yieldNum['text'] = round(boxStats.yield_rate, 2)
        common_error_list = self.most_common_error(boxStats)
        for i in range( len(common_error_list) ):
            self.commonErrorList[i]['text'] = f'{common_error_list[i][0]} : {common_error_list[i][1]} times'