# import dependencies
import time
import numpy as np
import serial
from harvesters.core import Harvester
from pathlib import Path
import csv
import cv2

def yesno(question):
    """Simple Yes/No Function."""
    prompt = f'{question} ? (y/n): '
    ans = input(prompt).strip().lower()
    if ans not in ['y', 'n']:
        print(f'{ans} is invalid, please try again...')
        return yesno(question)
    if ans == 'y':
        return True
    return False

# Get the date and time in yymmddHHMMSS format for file naming
Time = time.strftime("%y%m%d%H%M%S")

# create the directory for image saving
imageDir = Path("~/Particles/")/Time
# create the directory for image saving
imageDir.mkdir(parents=True,exist_ok=True)


print('opening gigE-vis platform')
# open the GeniCam API
h = Harvester()
# import the GenTL cti file
h.add_file('/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti')
# update Harvester to use the GenTL file
h.update()
# display the list of detected GigE cameras
print(h.device_info_list)

# create an interface to the camera
# activates cam, light green
ia = h.create(0)
# set the camera settings in the node_map
n=ia.remote_device.node_map
# n.AcquisitionMode.value = 'SingleFrame'
# n.TriggerSelector.value='AcquisitionStart'
n.acquisitionFrameRateControlMode.value = 'Programmable'
n.ExposureMode.value = 'Timed'
n.ExposureTime.value = 4004
n.AcquisitionFrameRate.value = 4
n.exposureAlignment.value = 'Synchronous'
n.Width.value = 4112
n.Height.value = 3008
n.autoBrightnessMode.value = 'Off'
n.devicePacketResendBufferSize.value = 0.1
n.DeviceLinkThroughputLimitMode.value	= 'On'
n.DeviceLinkThroughputLimit.value = 55000000 # I removed a 0 now it works
n.GevSCPSPacketSize.value	= 9000
n.GevStreamChannelSelector.value = 0

# create a function to communicate over Serial 
def query(serialPort,string):
    command="#1"+string+"\r"
    ser.write(bytes(command, 'utf-8'))
    time.sleep(0.1)
    stri = ser.read_until(b"\r").decode('UTF-8')[1:]
    # print(stri)
    return stri


ser = serial.Serial('/dev/ttyS4',baudrate=115200,
                         bytesize=serial.EIGHTBITS,
                         parity=serial.PARITY_NONE,
                         stopbits=serial.STOPBITS_ONE,
                         timeout=2000)
print('opening serial connection')
query(ser,"J0") # open connection
query(ser,"g32") # set 32nd step
query(ser,"p1") # set relative positioning
query(ser,":gn+200") # gear numerator
query(ser,"o+200") # set speed HZ
query(ser,":ramp_mode+2") # 
query(ser,":accel+50800") # set acceleration HZ/S
query(ser,":decel+50800") # set acceleration


# set the number of images per rev
n=40  
query(ser,"s+"+str(int(12800/n))) # step by total steps per rev/n

query(ser,"A")
print('Motor connected.')
time.sleep(3)
print(query(ser,"C"))
query(ser,"I")

#blank list for debugging
position=[]

ia.start(run_as_thread=True)

print('Camera ready.')

focusStacks = True
stack=0
while focusStacks==True:
    for idx,step in enumerate(np.arange(0,n-1)):
        # The buffer of the genie nano needs to be filled up before we can access the image, may not be required for other cameras
        for i in range(4):
            with ia.fetch() as buffer:
                component = buffer.payload.components[-1] #the buffer fills up, we want the latest entry, i.e. the last...

        buffer=ia.fetch()
        component = buffer.payload.components[-1] #the buffer fills up, we want the latest entry, i.e. the last...
        img = component.data.reshape(component.height, component.width)
        anglefolder = imageDir/(str(step).zfill(3))
        anglefolder.mkdir(parents=True,exist_ok=True)
        filename = anglefolder/(str(Time)+"_"+str(step).zfill(3)+"_fs"+str(stack).zfill(2)+".tiff")
        cv2.imwrite(str(filename),img)
        buffer.queue()

        #advance the motor
        query(ser,"A")
        time.sleep(3)
        query(ser,"C")
        position.append([query(ser,"I").strip(),idx,stack])
        print('Image {} of {}.'.format(idx+1,n,), sep=' ', end='\r', flush=True)

    stack+=1
    time.sleep(3)
    print('Finally, moving to theta zero')
    query(ser,"A")
    time.sleep(3)
    query(ser,"C")
    position.append([query(ser,"I").strip(),idx,stack])
    focusStacks=yesno('Stack {} complete. Continue imaging in a new stack'.format(stack))

print('Writing the debug file to parent directory...')

with open(str(imageDir/'debug.csv'), "w", newline="") as f:
    writer=csv.writer(f)
    writer.writerows(position)

print('Done!')
ia.destroy()
h.reset()

