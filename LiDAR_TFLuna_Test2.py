import serial,time
import sys
#
##########################
# TFLuna Lidar
##########################
#
ser = serial.Serial("/dev/serial0", 115200,timeout=0) # mini UART serial device
#
############################
# read ToF data from TF-Luna
############################
#
def read_tfluna_data():
    while True:
        counter = ser.in_waiting # count the number of bytes of the serial port
        if counter > 8:
            bytes_serial = ser.read(9) # read 9 bytes
            ser.reset_input_buffer() # reset buffer

            if bytes_serial[0] == 0x59 and bytes_serial[1] == 0x59: # check first two bytes
                distance = bytes_serial[2] + bytes_serial[3]*256 # distance in next two bytes
                return distance

if ser.isOpen() == False:
    ser.open() # open serial port if not open
    
try:
    while True:
        distance= read_tfluna_data() # read values
        print('Distance: {0:2.2f} cm'.format(distance)) # print sample data
except KeyboardInterrupt:
    print('Keyboard Interrupt')
except:
    eType = sys.exc_info()[0]  # Return exception type
    print(eType)
sys.exit()
ser.close() # close serial port
