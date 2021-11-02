import time
import multiprocessing
import TFLite_detection_webcam
import X

def machinelearning():
    exec(TFLite_detection_webcam)
def sensors():
    exec(

p1 = multiprocessing.Process(target=machinelearning(),args=[])
p2 = multiprocessing.Process(target=,args=[])
p3 = multiprocessing.Process(target=,args=[])

if __name__== '__main__':
    p1.start()
    p2.start()
    p3.start()
    
    p1.join()
    p2.join()
    p3.join()

finish = time.perf_counter()
print("Finished running after: ",finish)