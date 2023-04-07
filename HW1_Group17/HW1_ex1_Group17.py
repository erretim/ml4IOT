import argparse
import tensorflow as tf
import time, datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input file name', required=True)
parser.add_argument('--output', type=str, help='output file name', required=True)
parser.add_argument('--normalize', type=bool, default=False, help='option to normalize temperature', required=False)
args = parser.parse_args()

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

MIN_TEMP = 0
MAX_TEMP = 50

MIN_HUM = 20
MAX_HUM = 90

def convertFile():
    with tf.io.TFRecordWriter(args.output, 'GZIP') as writer:
        with open(args.input, 'r') as f:

            data_line = f.readline()

            while data_line != None and data_line != "":
                data_line = data_line.split(',')

                strdate = data_line[0] + ' ' + data_line[1]
                datetimeobj=datetime.datetime.strptime(strdate,"%d/%m/%Y %H:%M:%S")
                timeobj = int(time.mktime(datetimeobj.timetuple()))

                if args.normalize:
                  temp = (float(data_line[2]) - MIN_TEMP)/ (MAX_TEMP - MIN_TEMP)
                  hum = (float(data_line[3]) - MIN_HUM)/(MAX_HUM - MIN_HUM)

                  mapping = { 
                        'D': _int64_feature(timeobj), # <<< changed datatype
                        'T': _float_feature(temp), 
                        'H': _float_feature(hum)
                    }

                else:
                  temp = int(data_line[2])
                  hum = int(data_line[3])

                  mapping = { 
                        'D': _int64_feature(timeobj), # <<< changed datatype
                        'T': _int64_feature(temp), 
                        'H': _int64_feature(hum)
                  }
                
                example = tf.train.Example(features=tf.train.Features(feature=mapping))
                writer.write(example.SerializeToString())
                data_line = f.readline()


if __name__ == '__main__':
    convertFile()
    print(f'csv: {os.path.getsize(args.input)}B')
    print(f'tfrecord: {os.path.getsize(args.output)}B')