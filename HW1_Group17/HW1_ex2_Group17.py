import os
import tensorflow as tf
import time
import numpy as np
from subprocess import Popen


def MFCC_slow(file_path, frame_length, frame_step, num_mel_bins, sampling_rate, lower_frequency, upper_frequency, mel_coefs):
    Popen('sudo sh -c "echo performance >'
        '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"', 
        shell=True).wait()

    time_results =[]
    mfccs_results = []
    
    for i, filename in enumerate(os.listdir(file_path)):
        if not os.path.isdir(filename):
            start_time = time.time()

            #Read the audio signal
            audio = tf.io.read_file(f'{file_path}{filename}')

            #Convert the signal in a TensorFlow
            tf_audio, rate = tf.audio.decode_wav(audio)
            tf_audio = tf.squeeze(tf_audio, 1) #shape: (16000,) 
            
            # Convert the waveform in a spectrogram applying the STFT
            stft = tf.signal.stft(tf_audio, 
                        frame_length=frame_length, 
                        frame_step=frame_step,
                        fft_length=frame_length)
            spectrogram = tf.abs(stft) #shape: (49,321)
           
            if i == 0:
                #Compute the log-scaled Mel spectrogram
                num_spectrogram_bins = spectrogram.shape[-1]

                linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                        num_mel_bins,
                        num_spectrogram_bins,
                        sampling_rate,
                        lower_frequency,
                        upper_frequency)

            mel_spectrogram = tf.tensordot(
                        spectrogram, 
                        linear_to_mel_weight_matrix,
                        1)

            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6) #shape: (49,40)

            #Compute the MFCCs  #shape:(49,10)
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms( 
                        log_mel_spectrogram)[...,:mel_coefs]

            end_time = time.time()
            time_results.append(end_time-start_time)
            mfccs_results.append(mfccs)

    return time_results, mfccs_results

def MFCC_fast(file_path, frame_length, frame_step, num_mel_bins, sampling_rate, lower_frequency, upper_frequency, mel_coefs, factor):
    Popen('sudo sh -c "echo performance >'
        '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"', 
        shell=True).wait()

    time_results = []
    mfccs_results = []

    for i, filename in enumerate(os.listdir(file_path)):
        if not os.path.isdir(filename):

            start_time = time.time()
            #Read the audio signal
            audio = tf.io.read_file(f'{file_path}{filename}')
            #Convert the signal in a TensorFlow
            tf_audio, rate = tf.audio.decode_wav(audio)
            if factor > 1:
                tf_audio = tf.reshape(tf_audio, [int(sampling_rate/factor),factor])[:,0]
            else:
                tf_audio = tf.squeeze(tf_audio, 1) #shape: (16000,) 

        
            # Convert the waveform in a spectrogram applying the STFT
            stft = tf.signal.stft(tf_audio, 
                        frame_length=int(frame_length/factor), 
                        frame_step=int(frame_step/factor),
                        fft_length=int(frame_length/factor)
                        )   
            
            spectrogram = tf.abs(stft) 
            
            if i == 0:
                #Compute the log-scaled Mel spectrogram
                num_spectrogram_bins = spectrogram.shape[-1]

                linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                        num_mel_bins,
                        num_spectrogram_bins,
                        int(sampling_rate/factor),
                        lower_frequency,
                        int(upper_frequency/factor))

            mel_spectrogram = tf.tensordot(
                        spectrogram, 
                        linear_to_mel_weight_matrix,
                        1)

            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6) 
            
            #Compute the MFCCs  
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms( 
                        log_mel_spectrogram)[:,:mel_coefs]

            end_time = time.time()

            time_results.append(end_time-start_time)
            mfccs_results.append(mfccs)

    return time_results, mfccs_results



def getSNR(mfcc_listS, mfcc_listF):

    SNRs = []

    for mfccS, mfccF in zip(mfcc_listS, mfcc_listF):

        current_SNR = 20*np.log10((np.linalg.norm(mfccS))/(np.linalg.norm(mfccS - mfccF + 10e-6 )))
        SNRs.append(current_SNR)

    return np.mean(SNRs)


if __name__ == "__main__":
    
    file_path = './inputs/H1_yes_no/' #path to unziped files

    L = 1000 #ms
    l = 16 #ms
    s = 8 #ms

    rate = 16 #[samples/ms]

    frame_length = rate*l # rate [samples/ms] * 16 [ms] 
    frame_step = rate*s # rate [samples/ms] * 8 [ms] 

    num_mel_bins = 40
    sampling_rate = rate*1000
    lower_frequency = 20 #Hz
    upper_frequency = 4000 #Hz
    mel_coefs = 10 
    
    time_results_slow, mfccs_MFCC_slow = MFCC_slow(file_path, frame_length, frame_step, num_mel_bins, sampling_rate, lower_frequency, upper_frequency, mel_coefs)
    
    print(f'MFCC slow = {np.mean(time_results_slow)*1000:.2f} ms')

    # MFCC fast parameters
    factor = 2 # new_sf = old_sf/factor
    num_mel_bins = 32

    time_results_fast, mfccs_MFCC_fast = MFCC_fast(file_path, frame_length, frame_step, num_mel_bins, sampling_rate, lower_frequency, upper_frequency, mel_coefs, factor)
    
    print(f'MFCC fast = {np.mean(time_results_fast)*1000:.2f} ms')

    mean_SNR = getSNR(mfccs_MFCC_slow, mfccs_MFCC_fast)
    print(f'SNR = {mean_SNR:.2f} dB')