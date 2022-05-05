import time
import sys
import tfmplus as tfmP   # Import the `tfmplus` module v0.1.0
from tfmplus import *    # and command and paramter defintions

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

try:
    while True:
        time.sleep(0.05)   # Loop delay 50ms to match the 20Hz data frame rate
        # Use the 'getData' function to get data from device
        if( tfmP.getData()):
            print( f" Dist: {tfmP.dist:{3}}cm ", end= '')   # display distance
            print( f" | ", end= '\n')
        else:                  # If the command fails...
          tfmP.printFrame()    # display the error and HEX data
#
except KeyboardInterrupt:
    print( 'Keyboard Interrupt')
#    
except: # catch all other exceptions
    eType = sys.exc_info()[0]  # return exception type
    print( eType)
#
finally:
    print( "That's all folks!")
    sys.exit()                   # clean up the OS and exit