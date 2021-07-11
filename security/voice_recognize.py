import pyaudio
import wave
import cv2
import os
import pickle
import time
import aubio
from scipy.io.wavfile import read
from IPython.display import Audio, display, clear_output

from main_functions import *

def recognize():
    # Voice Authentication
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    FILENAME = "./test.wav"

    audio = pyaudio.PyAudio()
   
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    time.sleep(2.0)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")


    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # saving wav file 
    waveFile = wave.open(FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    modelpath = "./gmm_models/"

    gmm_files = [os.path.join(modelpath,fname) for fname in 
                os.listdir(modelpath) if fname.endswith('.gmm')]

    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]

    speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
                in gmm_files]
  
    if len(models) == 0:
        print("No Users in the Database!")
        return
        
    #read test file
    sr,audio = read(FILENAME)

    # extract mfcc features
    vector = extract_features(audio,sr)
    log_likelihood = np.zeros(len(models)) 

    #checking with each model one by one
    for i in range(len(models)):
        gmm = models[i]         
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    pred = np.argmax(log_likelihood)
    identity = speakers[pred]
   
    # if voice not recognized than terminate the process
    if identity == 'unknown':
            print("Not Recognized! Try again...")
            return
    
    print( "Recognized as - ", identity)

if __name__ == '__main__':
    BUFFER_SIZE             = 2048
    CHANNELS                = 1
    FORMAT                  = pyaudio.paFloat32
    METHOD                  = "default"
    SAMPLE_RATE             = 44100
    HOP_SIZE                = BUFFER_SIZE//2
    PERIOD_SIZE_IN_FRAME    = HOP_SIZE

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
    try:
        while True:
            data = mic.read(PERIOD_SIZE_IN_FRAME)
            # Convert into number that Aubio understand.
            samples = np.fromstring(data,dtype=aubio.float_type)
            # Finally get the pitch.
            pitch = pDetection(samples)[0]
            # Compute the energy (volume)
            # of the current frame.
            volume = np.sum(samples**2)/len(samples)
            #print(int(volume*1000))
            if int(volume*1000)>5:
                recognize()
    except KeyboardInterrupt:
        print("stopped")
        pass