import cherrypy
import json
import base64
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from scipy import signal


def res(audio, sampling_rate):        
    audio = signal.resample_poly(audio, 1, 16000 // sampling_rate)
    return np.array(audio, dtype = np.float32)

def tf_function(audio, sampling_rate):
    audio = tf.numpy_function(res, [audio, sampling_rate], tf.float32)
    return audio

def readFile(file):
    elems = []
    fp = open(file, 'r')
    for f in fp:
        elems.append(f.strip())
    return elems

class SignalPreprocessor:
    def __init__(self, labels, sampling_rate, frame_length, frame_step, num_mel_bins=None, lower_frequency=None, num_coefficients=None):
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
        # Transform the complex number in real number
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
        # resampling 
        if(self.sampling_rate != 16000):
            audio = tf_function(audio, self.sampling_rate)
        audio = tf.squeeze(audio, axis=1)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)
        return mfccs

    def PreprocessAudio(self, audio):
        data = self.preprocess(audio)
        return data

class CloudService(object):
    def __init__(self):
        L = 1000  # ms
        l = 40  # ms
        s = 20  # ms

        rate = 16  # [samples/ms]

        frame_length = rate*l  # rate [samples/ms] * 16 [ms]  #640
        frame_step = rate*s  # rate [samples/ms] * 8 [ms]	#320

        num_mel_bins = 40
        mel_coefs = 10
        sampling_rate = rate*1000

        special_characters = ['[', ']', '"', "'"]
        LABELS = readFile('labels.txt')[0]
        for s_c in special_characters:
            LABELS = LABELS.replace(s_c, "")
        LABELS = LABELS.split(',')
        LABELS = np.array([s.strip() for s in LABELS])


        self.sp = SignalPreprocessor(labels=LABELS, sampling_rate=sampling_rate, frame_length=frame_length, frame_step=frame_step,
                                     num_mel_bins=num_mel_bins, lower_frequency=20, num_coefficients=mel_coefs)

        model_path = "./kws_dscnn_True.tflite"

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    exposed = True

    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):

        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')

        body = cherrypy.request.body.read()
        body = json.loads(body)
        
        audio_str = body["e"][0]["vd"]

        if audio_str is None:
            raise cherrypy.HTTPError(400, 'Wrong query')

        audio_b64 = audio_str.encode()
        audio_binary = base64.urlsafe_b64decode(audio_b64)
        audio, _ = tf.audio.decode_wav(audio_binary)
        data = self.sp.PreprocessAudio(audio)
        data = tf.expand_dims(data, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], data)
        self.interpreter.invoke()
        y_pred = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        y_pred = int(np.argmax(y_pred))
        output = {
            'y_pred': y_pred
        }
        output_str = json.dumps(output)
        return output_str

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':

    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(CloudService(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8000})
    cherrypy.engine.start()
    cherrypy.engine.block()
