'''
HelloWorld example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.Session()

# Run the op
print(sess.run(hello))

'''
Create Annotation Subset
-Defines a subset of the speech command data set to be used for manual annotations
Robert Brais
'''

import input_data
import models

import os.path
import shutil

def main():
    wanted_words = 'yes,no,up,down,left,right,on,off,stop,go'
    sample_rate = 16000
    clip_duration_ms = 1000
    window_size_ms = 30.0
    window_stride_ms = 10.0
    dct_coefficient_count = 40

    data_url = ''
    data_dir = '/tmp/speech_dataset/'
    silence_percentage = 0
    unknown_percentage = 0
    validation_percentage = 1
    testing_percentage = 1

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()

    # Begin by making sure we have the training data we need. If you already have
    # training data of your own, use `--data_url= ` on the command line to avoid
    # downloading.
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(wanted_words.split(','))),
        sample_rate, clip_duration_ms, window_size_ms,
        window_stride_ms, dct_coefficient_count)
    audio_processor = input_data.AudioProcessor(
        data_url, data_dir, silence_percentage,
        unknown_percentage,
        wanted_words.split(','), validation_percentage,
        testing_percentage, model_settings)

    data, labels = audio_processor.get_unprocessed_data(-1, model_settings, 'testing')
    # print('CREATE ANNOTATION SUBSET: Printing data then labels.')
    # print(data)
    # print(labels)
    size = audio_processor.set_size('testing')
    print('CREATE ANNOTATION SUBSET: Printing annotation set size')
    print(size)

    annotation_listing = audio_processor.data_index['testing']
    print('CREATE ANNOTATION SUBSET: Printing annotation set names')
    print(annotation_listing)

    print('CREATE ANNOTATION SUBSET: Copying annotation subset to Projects folder')
    dest_directory = 'C:\\Users\\rober\\Documents\\Projects\\annotation_dataset\\'
    count = 0
    for annotation_details in annotation_listing:
        wav_path = annotation_details['file']
        dest_path = os.path.join(dest_directory,annotation_details['label'])
        shutil.copy(wav_path,dest_path)
        count = count + 1
    print(100.0 * count/size, '% of files copied')

if __name__ == "__main__":
    main()

