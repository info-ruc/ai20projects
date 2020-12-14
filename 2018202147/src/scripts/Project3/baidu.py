import sys
import json
import time
import os
import pyaudio
import wave
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode


class AudioMove():
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.scope = False
        self.token_url='http://openapi.baidu.com/oauth/2.0/token'
        
    def fetch_token(self):
        params = {'grant_type': 'client_credentials',
                'client_id': self.api_key,
                'client_secret': self.secret_key}
        post_data = urlencode(params)
        post_data = post_data.encode('utf-8')
        req = Request(self.token_url,post_data)
        try:
            f = urlopen(req)
            result_str = f.read()
        except URLError as err:
            print('token http response http code : ' + str(err.code))
            result_str = err.read()
        result_str = result_str.decode()
        #print(result_str)
        result = json.loads(result_str)
        #print(result)
        if ('access_token' in result.keys() and 'scope' in result.keys()):
            if self.scope and (not self.scope in result['scope'].split(' ')):  # self.scope = False 忽略检查
                print('scope is not correct')
                raise FileNotFoundError
            print('SUCCESS WITH TOKEN: %s ; EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
            return result['access_token']
        else:
            print('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')
            raise EOFError


    def transfer(self, audio_file,dev_pid=1537):
        timer = time.perf_counter
        token = self.fetch_token()
        audio_file = audio_file
        audio_format = audio_file[-3:]
        cuid = '123456PYTHON'
        rate = 16000
        dev_pid = dev_pid  #madarin
        asr_url='http://vop.baidu.com/server_api'
        speech_data = []
        with open(audio_file, 'rb') as speech_file:
            speech_data = speech_file.read()
        length = len(speech_data)
        if length == 0:
            raise EOFError

        params = {'cuid': cuid, 'token': token, 'dev_pid': dev_pid}
        params_query = urlencode(params)

        headers = {
            'Content-Type': 'audio/' + audio_format + '; rate=' + str(rate),
            'Content-Length': length
        }

        url = asr_url + "?" + params_query
        #print("url is", url)
        #print("header is", headers)
        # print post_data
        req = Request(asr_url + "?" + params_query, speech_data, headers)
        try:
            begin = timer()
            f = urlopen(req)
            result_str = f.read()
            print("Request time cost %f" % (timer() - begin))
        except  URLError as err:
            print('asr http response http code : ' + str(err.code))
            result_str = err.read()
        result_str = str(result_str, 'utf-8')
        print(result_str)

    def record_wav(self,filepath,seconds=5,channels=1):
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = channels
        fs = 16000  # Record at 44100 samples per second
        seconds = seconds
        filename = filepath

        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        print('Recording')

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []  # Initialize array to store frames

        # Store data in chunks for 3 seconds
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

        print('Finished recording')

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

    def play_audio(self,filepath):
        # Set chunk size of 1024 samples per data frame
        chunk = 1024  

        # Open the sound file 
        wf = wave.open(filepath, 'rb')

        # Create an interface to PortAudio
        p = pyaudio.PyAudio()

        # Open a .Stream object to write the WAV file to
        # 'output = True' indicates that the sound will be played rather than recorded
        stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                        channels = wf.getnchannels(),
                        rate = wf.getframerate(),
                        output = True)

        # Read data in chunks
        data = wf.readframes(chunk)

        # Play the sound by writing the audio data to the stream
        while data != '':
            stream.write(data)
            data = wf.readframes(chunk)

        # Close and terminate the stream
        stream.close()
        p.terminate()

    def realtime_recognize(self,seconds):
        temp = os.path.join(os.getcwd(), 'demowav.wav')
        self.record_wav(temp, seconds)
        self.transfer(temp)
        os.remove(temp)

    

if __name__ == "__main__":
    audiocar = AudioMove('yhg75Ddu4vPd6bk7qsHVDgEe', 'NEvYer9tH05MlaCqZHP6hYV2ICQINAOd')
    audiocar.realtime_recognize(5)


