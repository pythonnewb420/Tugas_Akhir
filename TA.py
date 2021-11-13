#!/usr/bin/env ptyhon
from threading import Thread, Lock
from time import sleep
import TFLite_detection_webcam
import VL53L0X
import Motor


# Global variable
tof = VL53L0X.VL53L0X()
distance = tof.get_distance()
mutex = Lock()
model_dir = coco_ssd_mobilenet_v1

# Define a function for the thread
def ml():
   # Create object from ObjectDetection
   detector = ObjectDetection()
   # looping
   while True:
    is_object_exist = detector.do_detection()
    print(f'ml: {is_object_exist} {distance}')
    sleep(3)

def lidar():
  global distance
  # looping
  while True:
    mutex.acquire()
    distance_object = distance
    mutex.release()
    is_obstacle = distance < 0.3
    print(f'lidar : {distance} {is_obstacle}')
    # kontrol: perlu informasi distance
    if is_object_exist and distance < 600:
      motor.stop()  # <-- condition to control motor
    if is_object_exist and distance < 1000:
      motor.forward()
    if is_object_exist and distance < 2000:
      motor.forward()
    sleep(1)
  
# Create two threads as follows
tf_object = tflite.TFLite()
tf_object.run()
tflite.run_outside()

try:
  t1 = Thread( target = ml, args=(model_dir) )
  t2 = Thread(target = lidar, args=())
  t1.start()
  t2.start()
except Exception as e:
  print ("Error: unable to start thread")
  print(e)
  
if __name__ == '__main__':     # Program start from here
    setup()
    try:
        loop()
    except KeyboardInterrupt:
        destroy()


