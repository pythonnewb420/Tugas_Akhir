# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import serial
import time
from gpiozero import Motor
import RPi.GPIO as GPIO
from time import sleep

#Create sensor and actuator object
#
############################
# Serial Functions
############################
#
#LiDAR
def read_tfluna_data():
    while True:
        counter = ser.in_waiting  # count the number of bytes waiting to be read
        bytes_to_read = 9
        if counter > bytes_to_read-1:
            bytes_serial = ser.read(bytes_to_read)  # read 9 bytes
            ser.reset_input_buffer()  # reset buffer

            if bytes_serial[0] == 0x59 and bytes_serial[1] == 0x59:  # check first two bytes
                # distance in next two bytes
                distance = bytes_serial[2] + bytes_serial[3]*256
                # signal strength in next two bytes
                strength = bytes_serial[4] + bytes_serial[5]*256
                # temp in next two bytes
                temperature = bytes_serial[6] + bytes_serial[7]*256
                temperature = (temperature/8) - 256  # temp scaling and offset
                return distance/100.0, strength, temperature


def set_samp_rate(samp_rate=100):
    ##########################
    # change the sample rate
    samp_rate_packet = [0x5a, 0x06, 0x03,
                        samp_rate, 00, 00]  # sample rate byte array
    ser.write(samp_rate_packet)  # send sample rate instruction
    return


def get_version():
    ##########################
    # get version info
    info_packet = [0x5a, 0x04, 0x14, 0x00]

    ser.write(info_packet)  # write packet
    time.sleep(0.1)  # wait to read
    bytes_to_read = 30  # prescribed in the product manual
    t0 = time.time()
    while (time.time()-t0) < 5:
        counter = ser.in_waiting
        if counter > bytes_to_read:
            bytes_data = ser.read(bytes_to_read)
            ser.reset_input_buffer()
            if bytes_data[0] == 0x5a:
                version = bytes_data[3:-1].decode('utf-8')
                print('Version -'+version)  # print version details
                return
            else:
                ser.write(info_packet)  # if fails, re-write packet
                time.sleep(0.1)  # wait


def set_baudrate(baud_indx=4):
    ##########################
    # get version info
    baud_hex = [[0x80, 0x25, 0x00],  # 9600
                [0x00, 0x4b, 0x00],  # 19200
                [0x00, 0x96, 0x00],  # 38400
                [0x00, 0xe1, 0x00],  # 57600
                [0x00, 0xc2, 0x01],  # 115200
                [0x00, 0x84, 0x03],  # 230400
                [0x00, 0x08, 0x07],  # 460800
                [0x00, 0x10, 0x0e]]  # 921600
    info_packet = [0x5a, 0x08, 0x06, baud_hex[baud_indx][0], baud_hex[baud_indx][1],
                   baud_hex[baud_indx][2], 0x00, 0x00]  # instruction packet

    prev_ser.write(info_packet)  # change the baud rate
    time.sleep(0.1)  # wait to settle
    prev_ser.close()  # close old serial port
    time.sleep(0.1)  # wait to settle
    ser_new = serial.Serial(
        "/dev/serial0", baudrates[baud_indx], timeout=0)  # new serial device
    if ser_new.isOpen() == False:
        ser_new.open()  # open serial port if not open
    bytes_to_read = 8
    t0 = time.time()
    while (time.time()-t0) < 5:
        counter = ser_new.in_waiting
        if counter > bytes_to_read:
            bytes_data = ser_new.read(bytes_to_read)
            ser_new.reset_input_buffer()
            if bytes_data[0] == 0x5a:
                indx = [ii for ii in range(0, len(baud_hex)) if
                        baud_hex[ii][0] == bytes_data[3] and
                        baud_hex[ii][1] == bytes_data[4] and
                        baud_hex[ii][2] == bytes_data[5]]
                print('Set Baud Rate = {0:1d}'.format(baudrates[indx[0]]))
                time.sleep(0.1)
                return ser_new
            else:
                ser_new.write(info_packet)  # try again if wrong data received
                time.sleep(0.1)  # wait 100ms
                continue


#
############################
# Configurations
############################
#
baudrates = [9600, 19200, 38400, 57600, 115200,
             230400, 460800, 921600]  # baud rates
prev_indx = 4  # previous baud rate index (current TF-Luna baudrate)
# mini UART serial device
prev_ser = serial.Serial("/dev/serial0", baudrates[prev_indx], timeout=0)
if prev_ser.isOpen() == False:
    prev_ser.open()  # open serial port if not open
baud_indx = 4  # baud rate to be changed to (new baudrate for TF-Luna)
ser = set_baudrate(baud_indx)  # set baudrate, get new serial at new baudrate
set_samp_rate(100)  # set sample rate 1-250
get_version()  # print version info for TF-Luna

#actuator setup
GPIO.setmode(GPIO.BOARD)
GPIO.setup(37, GPIO.OUT) #servo
GPIO.setup(13, GPIO.OUT) #motor

servo = GPIO.PWM(37, 50)
motor = GPIO.PWM(13, 50)
motor.start(0)
servo.start(0)

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=60):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

# Create window
cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    #Get distance measured by LiDAR
    distance = read_tfluna_data()
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Draw circle in center
            xcenter = xmin + (int(round((xmax - xmin) / 2)))
            ycenter = ymin + (int(round((ymax - ymin) / 2)))
            cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)

            # Print info
            print('Object ' + str(i) + ': ' + object_name + ' at (' + str(xcenter) + ', ' + str(ycenter) + ')')

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
    #Draw LiDAR Distance in corner of frame
    cv2.putText(frame,'Distance: {0:.2f} cm'.format(distance),(950, 700),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    
    #actuator action
    motor.ChangeDutyCycle(8)
    servo.ChangeDutyCycle(7.5)
    if distance < 100:
        servo.ChangeDutyCycle(5)
        motor.ChangeDutyCycle(6.3)
        time.sleep(5)
        motor.stop()
        servo.stop()

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
motor.stop()
servo.stop()
GPIO.cleanup()
cv2.destroyAllWindows()
videostream.stop()