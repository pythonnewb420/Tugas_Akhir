from threading import Thread, Lock
from time import sleep
import random
import tflite
from TFLite_detection_webcam import ObjectDetection
# Global variable
distance = 100
mutex = Lock()

# Define a function for the thread
def ml():
   # Create object from ObjectDetection
   detector = ObjectDetection()

   # looping
   while True:
    is_object_exist = detector.do_detection()
    print(f'ml: {is_object_exist} {distance}')
    # kontrol: perlu informasi distance
    if is_object_exist and distance < 0.3:
      print('STOP')
    sleep(3)

def lidar():
  global distance
  # looping
  while True:
    mutex.acquire()
    distance = random.random()
    mutex.release()
    is_obstacle = distance < 0.3
    print(f'lidar : {distance} {is_obstacle}')
    sleep(1)
  
# Create two threads as follows
tf_object = tflite.TFLite()
tf_object.run()
tflite.run_outside()

try:
  t1 = Thread( target = ml, args=() )
  t2 = Thread(target = lidar, args=())
  t1.start()
  t2.start()
except Exception as e:
  print ("Error: unable to start thread")
  print(e)
