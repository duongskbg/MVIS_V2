import serial, os
from serial import Serial

class RS232():
    def __init__(self):
        try:
            with open( os.path.join( os.getcwd(), 'Data', 'config.txt' ) ) as file:
                lines = file.readlines()
            comPort = lines[0][:-1]
            self.serialPort = Serial(port = comPort, baudrate = 9600, parity = serial.PARITY_NONE)
        except Exception as err:
            print(err)
    def send_data(self, data):
        try:
            if self.serialPort.isOpen():
                self.serialPort.write(data.encode())        
        except:
            pass
    def close_port(self):
        try:
            if self.serialPort.isOpen():
                self.serialPort.close()
        except:
            pass