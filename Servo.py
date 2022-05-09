import RPi.GPIO as GPIO
import time

# Pins for servo Driver Inputs and setup
# Raspberry Pi is told that pin index is the same as with the BOARD diagram not the BCM diagram
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(37, GPIO.OUT)  # GPIO pin 37 is set as output signal for the servo
servo = GPIO.PWM(37, 50)
# servo initial state is OFF because the .start() is given 0 volts
servo.start(0)

#Speed Calibration for Brushed DC servo using PWM and ChangeDutyCycle
cdc = 5  #Servo is set so that the rudder angle will be at MAX starboard angle
servo.ChangeDutyCycle(cdc)
print(cdc)
time.sleep(1)  # servo is shut OFF after 5s usage

#command to stop the servo
servo.stop()
GPIO.cleanup()