import aubio
import numpy as num
import pyaudio
import sys
import cv2

# Some constants for setting the PyAudio and the
# Aubio.
BUFFER_SIZE             = 2048
CHANNELS                = 1
FORMAT                  = pyaudio.paFloat32
METHOD                  = "default"
SAMPLE_RATE             = 44100
HOP_SIZE                = BUFFER_SIZE//2
PERIOD_SIZE_IN_FRAME    = HOP_SIZE



    # Initiating PyAudio object.
pA = pyaudio.PyAudio()
    # Open the microphone stream.
mic = pA.open(format=FORMAT, channels=CHANNELS,rate=SAMPLE_RATE, input=True,frames_per_buffer=PERIOD_SIZE_IN_FRAME)

    # Initiating Aubio's pitch detection object.
pDetection = aubio.pitch(METHOD, BUFFER_SIZE,HOP_SIZE, SAMPLE_RATE)
    # Set unit.
pDetection.set_unit("Hz")
    # Frequency under -40 dB will considered
    # as a silence.
pDetection.set_silence(-40)

    # Infinite loop!
while True:

        # Always listening to the microphone.
    data = mic.read(PERIOD_SIZE_IN_FRAME)
        # Convert into number that Aubio understand.
    samples = num.fromstring(data,
            dtype=aubio.float_type)
        # Finally get the pitch.
    pitch = pDetection(samples)[0]
        # Compute the energy (volume)
        # of the current frame.
    volume = num.sum(samples**2)/len(samples)
        # Format the volume output so it only
        # displays at most six numbers behind 0.
    #volume = "{:6f}".format(volume)
    #volume=volume*100000
        # Finally print the pitch and the volume.
    print( int(volume*1000))

        

