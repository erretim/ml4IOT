import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import zlib
import argparse


parser = argparse.ArgumentParser() 
parser.add_argument('--version', type=str, required=True, help='a or b')
args = parser.parse_args()


#setup a random seed
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


zip_path = tf.keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip", 
    fname='jena_climate_2009_2016.csv.zip', 
    extract=True, 
    cache_dir='.', cache_subdir='data')


csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)
column_indices = [2,5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)


n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]


mean = train_data.mean(axis=0)
std = train_data.std(axis=0)


class WindowGenerator:
    def __init__(self, input_width, label_width, num_features, mean, std):
        self.input_width = input_width
        self.label_width = label_width
        self.num_features = num_features
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        # features -> set of sequences made of input_width + label_width values each. [#batch, (input+label)_width, 2] 
        inputs = features[:, :-self.label_width, :]
        labels = features[:, -self.label_width:, :]

        inputs.set_shape([None, self.input_width, self.num_features])
        labels.set_shape([None, self.label_width, self.num_features])
        
        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, reshuffle):
        # Creates a dataset of sliding windows over a timeseries provided as array
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data, # consecutive data points
                targets=None, # None -> the dataset will only yield the input data
                sequence_length=self.input_width + self.label_width, # Length of the output sequences
                sequence_stride=1, # Period between successive output sequences
                batch_size=32) # Number of timeseries samples in each batch 
        
        # from each set of sequences it splits data to get input and labels and then normalize
        ds = ds.map(self.preprocess)

        # so the mapping is done only once
        ds = ds.cache()
        if reshuffle:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


class MultiOutputMAE(tf.keras.metrics.Metric):

    def __init__(self, name='mean_absolute_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=(2,))
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None): 
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=[0,1])
        self.total.assign_add(error)
        self.count.assign_add(1.)
        return
    
    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
    
    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)
        return result


input_width = 6
num_features = 2

if args.version == 'a':
    label_width = 3
elif args.version == 'b':
    label_width = 9
else:
    print('version Error!')
    exit()

generator = WindowGenerator(input_width, label_width, num_features, mean=mean, std=std)
train_ds = generator.make_dataset(train_data, reshuffle=True)
val_ds = generator.make_dataset(val_data, reshuffle=False)
test_ds = generator.make_dataset(test_data, reshuffle=False)


if args.version == 'a':

    alpha = 0.04
    sparsity=0.9

    base_model = tf.keras.Sequential([ 
        # tf.keras.layers.Conv1D(filters=int(64*alpha), kernel_size=(3,), activation='relu'), 
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(units=int(64*alpha), activation='relu'), 
        tf.keras.layers.Dense(units=num_features*label_width),
        tf.keras.layers.Reshape([label_width, num_features])
    ])

else:
    alpha = 0.04
    sparsity=0.9

    base_model = tf.keras.Sequential([ 
        tf.keras.layers.Conv1D(filters=int(64*alpha), kernel_size=(3,), activation='relu'), 
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(units=int(64*alpha), activation='relu'), 
        tf.keras.layers.Dense(units=num_features*label_width),
        tf.keras.layers.Reshape([label_width, num_features])
    ])

loss = tf.losses.MeanSquaredError()
optimizer = tf.optimizers.Adam()
metrics = [MultiOutputMAE()] 


checkpoint_filepath = f'./checkpoints/base_{label_width}/'

if not os.path.exists(checkpoint_filepath):
    os.makedirs(checkpoint_filepath)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min', #based on the mean_squared_error results the checkpoint will store the model with the lowest error.
    save_best_only=True)


base_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

base_model.fit(train_ds,   
    epochs=20, 
    validation_data=val_ds,
    callbacks=[model_checkpoint_callback]
    )

base_model.load_weights(checkpoint_filepath)


# Pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
	'pruning_schedule':
		tfmot.sparsity.keras.PolynomialDecay(
		initial_sparsity=0.30,
		final_sparsity=sparsity,
		power=3,
		begin_step=len(train_ds)*3,
		end_step=len(train_ds)*15)}


model_for_pruning = prune_low_magnitude(base_model, **pruning_params)

model_sparcity_callback = tfmot.sparsity.keras.UpdatePruningStep()


model_for_pruning.compile(loss=loss, optimizer=optimizer, metrics=metrics)

checkpoint_filepath = f'./checkpoints/prun_{label_width}/'

if not os.path.exists(checkpoint_filepath):
    os.makedirs(checkpoint_filepath)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min', #based on the mean_squared_error results the checkpoint will store the model with the lowest error.
    save_best_only=True)

model_for_pruning.fit(train_ds,   
    epochs=20, 
    validation_data=val_ds, 
    callbacks=[model_sparcity_callback, model_checkpoint_callback]
    )

model_for_pruning.load_weights(checkpoint_filepath)


model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_buffer = converter.convert()
tflite_compressed = zlib.compress(tflite_buffer)

MODEL_FILE_NAME = f'Group17_th_{args.version}.tflite.zlib'

if not os.path.exists(f'./models/'):
    os.makedirs(f'./models/')

saved_model_dir = os.path.join(f'./models/', MODEL_FILE_NAME)

with open(saved_model_dir, 'wb') as f:
    f.write(tflite_compressed)

print('File size: ' + str(round(os.path.getsize(saved_model_dir)/1024, 4)) + ' Kilobytes')

if not os.path.exists(f'./th_test_{args.version}'):
    os.makedirs(f'./th_test_{args.version}')
    tf.data.experimental.save(test_ds, f'./th_test_{args.version}')