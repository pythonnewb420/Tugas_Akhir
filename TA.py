#!/usr/bin/env ptyhon3
# import Motor


from threading import Thread, Lock
import TFLite_detection_webcam
from time import sleep
import time
import VL53L0X

# Create a VL53L0X object
tof = VL53L0X.VL53L0X()
#mutex = Lock()

def ml():
   # Create object from ObjectDetection
   detector = ObjectDetection()
   # looping
   while True:
    is_object_exist = detector.do_detection()
    print(f'ml: {is_object_exist} {distance}')
    sleep(3)

# Start ranging
def lidar():
  tof.start_ranging(VL53L0X.VL53L0X_BETTER_ACCURACY_MODE)
  timing = tof.get_timing()
  if (timing < 20000):
      timing = 20000
  print ("Timing %d ms" % (timing/1000))

  while True:
      distance = tof.get_distance()
      if (distance > 10):
          print ("%d cm %d" % ( (distance/10), "Tidak Aman"))

      time.sleep(timing/1000000.00)

# Create two threads as follows
tf_object = tflite.TFLite()
tf_object.run()
tflite.run_outside()

try:
  t1 = Thread(target = ml, args=())
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