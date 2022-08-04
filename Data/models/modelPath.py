import os

class mGetPath():
    dataPath = os.path.join(os.getcwd(), 'Data')
    classesPath = os.path.join( os.getcwd(), 'Data', 'classes.txt' ) 
    modelPath = os.path.join( os.getcwd(), 'Data', 'best.pt' )
    licensePath = os.path.join( os.getcwd(), 'Data', 'License', 'license.lic' ) 
    iniPath = os.path.join( os.getcwd(), 'Data', 'PyCV.ini' )  
    configPath = os.path.join( os.getcwd(), 'Data', 'camera_config.txt' )  
    iconPath = os.path.join( os.getcwd(), 'Data', 'app_icon.png' )  
    videoPath = os.path.join( dataPath, 'UDMP00SD.avi' )
    logPath = os.path.join( dataPath, 'log.txt')
    log2Path = os.path.join( dataPath, 'daily_log.txt')