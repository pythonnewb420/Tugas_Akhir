import RPi.GPIO as GPIO
from time import sleep, time

# Pins for Motor Driver Inputs and setup
GPIO.setmode(GPIO.BOARD) #Raspberry Pi is told that pin index is the same as with the BOARD diagram not the BCM diagram
GPIO.setup(13, GPIO.OUT) #GPIO pin 13 is set as output signal for the motor
motor = GPIO.PWM(13, 50)
motor.start(0) #Motor initial state is OFF because the .start() is given 0 volts

#Speed Calibration for Brushed DC Motor using PWM and ChangeDutyCycle
cdc = 5 #5 as in 5% of the total 100% of full PWM
motor.ChangeDutyCycle(5)
time.sleep(5) #motor is shut OFF after 5s usage

#command to stop the motor
motor.stop()
GPIO.cleanup()