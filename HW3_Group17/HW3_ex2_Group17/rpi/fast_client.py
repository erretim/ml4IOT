import tensorflow as tf
import numpy as np
import os
from scipy import signal
import base64
import requests
import time
from datetime import datetime
import json


def resampling_scipy(audio, sampling_rate):        
    audio = signal.resample_poly(audio, 1, 16000 // sampling_rate)
    return np.array(audio, dtype = np.float32)

def resampling_tf(audio, sampling_rate):
    audio = tf.numpy_function(resampling_scipy, [audio, sampling_rate], tf.float32)
    return audio

class SignalPreprocessor:
    def __init__(self, labels, sampling_rate, frame_length, frame_step, num_mel_bins=None, lower_frequency=None, upper_frequency=None, num_coefficients=None):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = sampling_rate//2
        self.num_coefficients = num_coefficients

        num_spectrogram_bins = self.frame_length // 2 + 1
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins,
            num_spectrogram_bins,
            int(self.sampling_rate),
            self.lower_frequency,
            int(self.upper_frequency))
        self.preprocess = self.preprocess_with_mfcc

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        return audio_binary, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio,
                              frame_length=int(self.frame_length),
                              frame_step=int(self.frame_step),
                              fft_length=int(self.frame_length))
        spectrogram = tf.abs(stft)
        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                                       self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[:, :self.num_coefficients]
        return mfccs

    def preprocess_with_mfcc(self, audio):
        if(self.sampling_rate != 16000):
            audio = resampling_tf(audio, self.sampling_rate)
        audio = tf.squeeze(audio, axis=1)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)
        return mfccs

    def PreprocessAudio(self, audio):
        data = self.preprocess(audio)
        return data


def readFile(file):
    elems = []
    fp = open(file, 'r')
    for f in fp:
        elems.append(f.strip())
    return elems


def LoadData(LABELS):
    # Download and extract the .csv file. The result is cached to avoid to download everytime
    zip_path = tf.keras.utils.get_file(
        origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

    data_dir = os.path.join('.', 'data', 'mini_speech_commands')

    test_files = readFile('kws_test_split.txt')

    L = 1000  # ms
    l = 40  # ms
    s = 20  # ms

    rate = 8  # [samples/ms]

    frame_length = rate*l  # rate [samples/ms] * 40 [ms]  #640
    frame_step = rate*s  # rate [samples/ms] * 20 [ms]	#320

    num_mel_bins = 16
    mel_coefs = 10
    sampling_rate = rate*1000

    sp = SignalPreprocessor(labels=LABELS, sampling_rate=sampling_rate, frame_length=frame_length, frame_step=frame_step,
                            num_mel_bins=num_mel_bins, lower_frequency=20, upper_frequency=4000, num_coefficients=mel_coefs)

    return sp, test_files


if __name__ == '__main__':

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    url = f"http://10.4.0.189:8000/"

    special_characters = ['[', ']', '"', "'"]
    LABELS = readFile('labels.txt')[0]
    for s_c in special_characters:
        LABELS = LABELS.replace(s_c, "")
    LABELS = LABELS.split(',')
    LABELS = np.array([s.strip() for s in LABELS])

    sp, test_files = LoadData(LABELS)

    model_path = "./kws_dscnn_True.tflite"

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    comm_size = 0
    accuracy = 0
    count = 0
    difs = []
    times = []
    
    for t in test_files:
        audio_binary, label = sp.read(t)
        audio, _ = tf.audio.decode_wav(audio_binary)

        start_time = time.time()
        
        data = sp.PreprocessAudio(audio)
        data = tf.expand_dims(data, axis=0)
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]['index'])
        
        end_time = time.time()
        
        y_pred = y_pred.squeeze()
        best_two = y_pred[np.argsort(y_pred)[-2:]]
        diff = abs(best_two[0] - best_two[1])
        difs.append(diff)
        

        if diff <= 0.25:
            audio_binary = audio_binary.numpy()
            audio_b64 = base64.urlsafe_b64encode(audio_binary)
            audio_str = audio_b64.decode()

            timestamp = int(datetime.now().timestamp())

            msg = {
                'bn': 'rpi', 
                'bt': timestamp,
                'e': [
                    {'n': 'audio',
                     'u': '/', 
                     't': 0, 
                     'vd': audio_str}
                ]
            }

            msg_str = json.dumps(msg)

            comm_size += len(msg_str)

            r = requests.post(url, json=msg)

            if r.status_code == 200:
                y_pred = int(np.array(int(r.json()['y_pred'])))
            else:
                print('Error:', r.status_code)
                exit()

        else:
            y_pred = int(np.argmax(y_pred))
        
        
        if y_pred == int(label):
            accuracy += 1

        count += 1

        times.append(end_time-start_time)

    print(f'Total accuracy: {(accuracy/count):.2f} %')
    print(f'Average execution time: {(np.average(times)*1000):.2f} ms')
    print(f'Total communication costs:  {(comm_size/(1024**2)):.2f} MB')
