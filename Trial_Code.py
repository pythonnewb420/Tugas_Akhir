import serial,time
import sys
import RPi.GPIO as GPIO
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

# Pins for Driver Inputs and setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD) #Raspberry Pi is told that pin index is the same as with the BOARD diagram not the BCM diagram
GPIO.setup(13, GPIO.OUT) #GPIO pin 13 is set as output signal for the motor
GPIO.setup(37, GPIO.OUT)  # GPIO pin 37 is set as output signal for the servo


servo = GPIO.PWM(37, 50)
motor = GPIO.PWM(13, 50)
motor.start(0) #Motor initial state is OFF because the .start() is given 0 volts
servo.start(0) #Servo initial state is OFF because the .start() is given 0 volts
#Speed and anhlr Calibration for PWM and ChangeDutyCycle
cdc_motor = 6.5  # 5 as in 5% of the total 100% of full PWM
cdc_Servo = 5  # Servo is set so that the rudder angle will be at MAX starboard angle

if ser.isOpen() == False:
    ser.open() # open serial port if not open
    
try:
    while True:
        distance= read_tfluna_data() # read values
        servo.ChangeDutyCycle(7.5)
        motor.ChangeDutyCycle(6.5)
        print('Distance: {0:2.2f} cm'.format(distance)) # print sample data
        if (distance < 120):
                servo.ChangeDutyCycle(cdc_Servo)
                motor.ChangeDutyCycle(cdc_motor)
                time.sleep(5)
                break
        else:
            continue
except KeyboardInterrupt:
    print('Keyboard Interrupt')
except:
    eType = sys.exc_info()[0]  # Return exception type
    print(eType)
motor.stop()
servo.stop()
GPIO.cleanup()
sys.exit()
ser.close() # close serial port

