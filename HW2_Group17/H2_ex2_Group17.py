import os
import numpy as np
import tensorflow as tf
import zlib
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='a, b or c')
args = parser.parse_args()

# random seeds
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


# data downloading
zip_path = tf.keras.utils.get_file(
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')
        
data_dir = os.path.join('.', 'data', 'mini_speech_commands')


def readFile(file):
  elems = []
  fp = open(file,'r')
  for f in fp:
    if f != '':
      elems.append(f)
    
  return elems

train_files= tf.random.shuffle(readFile('.\splitFiles\kws_train_split.txt'))
val_files = tf.random.shuffle(readFile('.\splitFiles\kws_val_split.txt'))
test_files =tf.random.shuffle(readFile('.\splitFiles\kws_test_split.txt'))

labels = open("labels.txt", "r")
labels = str(labels.read())
characters_to_remove = "[]''""  "
for character in characters_to_remove: 
    labels = labels.replace(character, "")
LABELS = labels.split(",")
n_output = len(LABELS)

# Signal generator
class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1  #integer division by 2 plus 1.

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep) # ie. [., data, mini_speech_commands, down, 9dc1889e_nohash_1.wav]
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    # if audio file is shorter than 1sec it is necessary to zero-pad it
    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    #calculates the abs(stft) 'spectrogram'
    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    # obtains the mel coefs from a spectrogram
    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs
    
    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio) # [w, h] = [124, 129]
        spectrogram = tf.expand_dims(spectrogram, -1)  #[w, h, 1] (3-D Tensor of shape [height, width, channels]).
        spectrogram = tf.image.resize(spectrogram, [32, 32])  #[32, 32, 1] (esentially the same data but "pixeled" shrinking the size in H & W)

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio) # [w,h] = [49, 321]
        mfccs = self.get_mfccs(spectrogram)  #[w, 10] = [49, 10]
        mfccs = tf.expand_dims(mfccs, -1) #[w, 10, 1] (adds batch dimension)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files) #makes each audiofile a tensor
        ds = ds.map(self.preprocess, num_parallel_calls=4) # num_parallel_calls makes a threadpool for GPU 
        ds = ds.batch(32) # number of consecutive elements of this dataset to combine in a single batch
        ds = ds.cache() # will produce exactly the same elements during each iteration through the dataset. Otherwise every call will map again the values.
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True) # randomize the cache dataset 

        return ds

options = {'frame_length': 256, 'frame_step': 128, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}
strides = [2, 1]


generator = SignalGenerator(LABELS, 16000, **options) # ** any other named parameters
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)

if args.version == 'a':

    alpha = 0.25

elif args.version == 'b':
    
    alpha = 0.23 

else:

    alpha = 0.15

dscnn = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=int(256*alpha),kernel_size=[3,3],strides=strides,use_bias=False), #input_shape=(32,32,1)
      tf.keras.layers.BatchNormalization(momentum=0.1),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.DepthwiseConv2D(kernel_size=[2,2], strides=[1,1], use_bias=False),
      tf.keras.layers.Conv2D(filters=int(256*alpha),kernel_size=[2,2],strides=[1,1],use_bias=False),
      tf.keras.layers.BatchNormalization(momentum=0.1),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.DepthwiseConv2D(kernel_size=[2,2], strides=[1, 1], use_bias=False),
      tf.keras.layers.Conv2D(filters=int(256*alpha),kernel_size=[2,2],strides=[1,1],use_bias=False),
      tf.keras.layers.BatchNormalization(momentum=0.1),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.DepthwiseConv2D(kernel_size=[2,2], strides=[1, 1], use_bias=False),
      tf.keras.layers.Conv2D(filters=int(256*alpha),kernel_size=[2,2],strides=[1,1],use_bias=False),
      tf.keras.layers.BatchNormalization(momentum=0.1),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(units=n_output)
      ])

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()
metrics = ['sparse_categorical_accuracy']


# checkpoint on best model
checkpoint_filepath = f'./checkpoints/base/'
if not os.path.exists(checkpoint_filepath):
    os.makedirs(checkpoint_filepath)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_sparse_categorical_accuracy',
    mode='max', #based on the mean_squared_error results the checkpoint will store the model with the lowest error.
    save_best_only=True)

base_model = dscnn
base_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = base_model.fit( 
    train_ds,
    batch_size=32, 
    epochs=40, 
    validation_data=val_ds,
    callbacks=[model_checkpoint_callback]
)

base_model.load_weights(checkpoint_filepath)


# quantization 

model_for_export = base_model

def representative_dataset_gen():
    for x, _ in train_ds.take(1000):
        yield [x]

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

if args.version == 'b' or args.version == 'c':
    converter.representative_dataset = representative_dataset_gen
elif args.version == 'a':
    converter.target_spec.supported_types = [tf.float16]

tflite_buffer = converter.convert()
tflite_compressed = zlib.compress(tflite_buffer)


MODEL_FILE_NAME = f'Group17_kws_{args.version}.tflite.zlib'
if not os.path.exists(f'./models/'):
    os.makedirs(f'./models/')
saved_model_dir = os.path.join(f'./models/', MODEL_FILE_NAME)
with open(saved_model_dir, 'wb') as f:
    f.write(tflite_compressed)

print('File size: ' + str(round(os.path.getsize(saved_model_dir)/1024, 4)) + ' Kilobytes')

if not os.path.exists(f'./th_test'):
    os.makedirs(f'./th_test')
    tf.data.experimental.save(test_ds, './th_test')


# Accuracy evaluation usually done on edge side.

import time

TEST_DIR = './th_test/'
with open(saved_model_dir, 'rb') as fp:
    decomp_model = zlib.decompress(fp.read())
    decomp_model_path = saved_model_dir[:-4] # .tf file
    
    file = open(decomp_model_path,'wb')
    file.write(decomp_model)
    file.close()

test_ds1 = test_ds.unbatch().batch(1)
interpreter = tf.lite.Interpreter(model_path=decomp_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
acc = 0
n = 0
time_infe = 0

for x,y in test_ds1:
  input_data = x
  y_true = y.numpy()[0]
  
  ti = time.time()
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  my_output = interpreter.get_tensor(output_details[0]['index'])[0]
  time_infe += time.time()-ti

  n+=1
  index_pred = np.argmax(my_output)
  if(index_pred==y_true):
    acc += 1

print(f'Accuracy: {(acc/n):.3f}, time: {(time_infe/n)*1000} ms')