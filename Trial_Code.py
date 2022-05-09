import RPi.GPIO as GPIO
import time
import sys
import tfmplus as tfmP   # Import the `tfmplus` module v0.1.0
from tfmplus import *    # and command and paramter defintions

#LiDAR Setup
serialPort = "/dev/serial0"  # Raspberry Pi normal serial port
serialRate = 115200          # TFMini-Plus default baud rate

# - - - Set and Test serial communication - - - -
print( "Serial port: ", end= '')
if( tfmP.begin( serialPort, serialRate)):
    print( "ready.")
else:
    print( "not ready")
    sys.exit()   #  quit the program if serial not ready

# - - Perform a system reset - - - - - - - -
print( "Soft reset: ", end= '')
if( tfmP.sendCommand( SOFT_RESET, 0)):
    print( "passed.")
else:
    tfmP.printReply()
# - - - - - - - - - - - - - - - - - - - - - - - -
time.sleep(0.5)  # allow 500ms for reset to complete
# - - - - - - - - - - - - - - - - - - - - - - - -

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

try:
    while True:
        time.sleep(0.05)   # Delay 50ms for 20Hz loop-rate
        if( tfmP.getData()):
            print( f" Dist: {tfmP.dist:{3}}cm ", end= '')   # display distance
            print( f" | ", end= '\n')
            if (tfmP.dist < 120):
                servo.ChangeDutyCycle(cdc_Servo)
                motor.ChangeDutyCycle(cdc_motor)
            else:
                continue
        else:                  # If the command fails...
            tfmP.printFrame()    # display the error and HEX data
            motor.stop()
            servo.stop()
            GPIO.cleanup()
            break
    #
    #  Use control-C to break loop
except KeyboardInterrupt:
    print('Keyboard Interrupt')
    #command to stop the motor
    motor.stop()
    servo.stop()
    GPIO.cleanup()
    #
    #  Catch all other exceptions
except:
    eType = sys.exc_info()[0]  # Return exception type
    print(eType)
print("That's all folks!")  # Say "Goodbye!"
sys.exit()
