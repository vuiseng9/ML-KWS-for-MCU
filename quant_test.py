# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modifications Copyright 2017-2018 Arm Inc. All Rights Reserved. 
# Adapted from freeze.py to run quantized inference on train/val/test dataset on the 
# trained model in the form of checkpoint
#          
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

import tensorflow as tf
import input_data
import quant_models as models
from glob import glob
from datetime import datetime
from generate_activation_stats import eventfile_to_df
from collections import OrderedDict

def derive_qformat(input_value, total_bits=7):
    # assumption: first bit is signed bit, meaning actual total bit is total_bits + 1

    # max_value must be positive
    max_value = abs(input_value)

    max_without_fracbit = 2**total_bits - 1
    if max_value > max_without_fracbit:
        # saturate/clip input value to maximum 
        max_value = max_without_fracbit

    int_bits = int(np.ceil(np.log2(max_value)))
    if int_bits < 0:
        int_bits = 0 # int_bits will be negative when the max(abs()) is lower than 1
    
    frac_bits = total_bits-int_bits

    # calculate Q format range
    qformat_min = -1 * (2**int_bits)
    qformat_max = (2**int_bits) - (2**-frac_bits)

    # print("input: {}; \nQ: Q{}.{}; \nQ.min: {}; \nQ.max: {}".format(
    #     input_value,
    #     int_bits,
    #     frac_bits,
    #     qformat_min,
    #     qformat_max
    # ))
    return input_value, int_bits, frac_bits, qformat_min, qformat_max

def suggest_bias_and_out_shift(weight_qformat, act_stats):
    # We choose mean of max value in each activation (other option is global max but it might be too extreme)
    # why? specific to DNN here, the relu layer will filter out all negative range value to zero
    # what we want is to retain the positive range value
    print('_'*100+"\n")
    input_qformat = derive_qformat(act_stats['input/Max_0']['mean'])
    _, current_Qm, current_Qn, current_qmin, _ = input_qformat

    # first fc layer
    # ------
    # q format after matrix multiplication (matvec)
    print("{:25} | Q{:2}.{} | act_max: {}".format('fc1 input', current_Qm, current_Qn, -1*(current_qmin)))
    w_Qm = weight_qformat['fc1_W_0']['m']
    w_Qn = weight_qformat['fc1_W_0']['n']
    print("{:25} | Q{:2}.{}".format('fc1 weight', w_Qm, w_Qn))
    current_Qm += w_Qm
    current_Qn += w_Qn
    print("{:25} | Q{:2}.{}".format('post input*weight', current_Qm, current_Qn))
    b_Qm = weight_qformat['fc1_b_0']['m']
    b_Qn = weight_qformat['fc1_b_0']['n']
    print("{:25} | Q{:2}.{}".format('fc1 bias', b_Qm, b_Qn))
    bias_shift = current_Qn - b_Qn
    assert bias_shift>=0, "this is a rare case, pls debug bias_shift"
    print("{:25} | Qn from {} to {}".format('bias shift (left)', b_Qn, current_Qn))

    fc2_in_qformat = derive_qformat(act_stats['fc1/fc1/Max_0']['mean'])
    _, next_Qm, next_Qn, _, _ = fc2_in_qformat
    print("{:25} | Q{}.{} -> Q{}.{}".format(
        'fc1_out -> fc2_input', 
        current_Qm, current_Qn,
        next_Qm, next_Qn))
    out_shift = current_Qn - next_Qn 
    assert out_shift>=0, "this is a rare case, pls debug out_shift"
    print("{:25} | Qn from {} to {}".format('out shift (right)', current_Qn, next_Qn))
    print('-'*100)
    print("fc1 | bias_shift = {} | out_shift = {}".format(bias_shift, out_shift))
    print('_'*100+"\n")

    # fc2
    #---
    _, current_Qm, current_Qn, current_qmin, _ = fc2_in_qformat
    print("{:25} | Q{:2}.{} | act_max: {}".format('fc2 input', current_Qm, current_Qn, -1*(current_qmin)))
    w_Qm = weight_qformat['fc2_W_0']['m']
    w_Qn = weight_qformat['fc2_W_0']['n']
    print("{:25} | Q{:2}.{}".format('fc2 weight', w_Qm, w_Qn))
    current_Qm += w_Qm
    current_Qn += w_Qn
    print("{:25} | Q{:2}.{}".format('post input*weight', current_Qm, current_Qn))
    b_Qm = weight_qformat['fc2_b_0']['m']
    b_Qn = weight_qformat['fc2_b_0']['n']
    print("{:25} | Q{:2}.{}".format('fc2 bias', b_Qm, b_Qn))
    bias_shift = current_Qn - b_Qn
    assert bias_shift>=0, "this is a rare case, pls debug bias_shift"
    print("{:25} | Qn from {} to {}".format('bias shift (left)', b_Qn, current_Qn))

    fc3_in_qformat = derive_qformat(act_stats['fc2/fc2/Max_0']['mean'])
    _, next_Qm, next_Qn, _, _ = fc3_in_qformat
    print("{:25} | Q{}.{} -> Q{}.{}".format(
        'fc2_out -> fc3_input', 
        current_Qm, current_Qn,
        next_Qm, next_Qn))
    out_shift = current_Qn - next_Qn 
    assert out_shift>=0, "this is a rare case, pls debug out_shift"
    print("{:25} | Qn from {} to {}".format('out shift (right)', current_Qn, next_Qn))
    print('-'*100)
    print("fc2 | bias_shift = {} | out_shift = {}".format(bias_shift, out_shift))
    print('_'*100+"\n")

    # fc3
    #---
    _, current_Qm, current_Qn, current_qmin, _ = fc3_in_qformat
    print("{:25} | Q{:2}.{} | act_max: {}".format('fc3 input', current_Qm, current_Qn, -1*(current_qmin)))
    w_Qm = weight_qformat['fc3_W_0']['m']
    w_Qn = weight_qformat['fc3_W_0']['n']
    print("{:25} | Q{:2}.{}".format('fc3 weight', w_Qm, w_Qn))
    current_Qm += w_Qm
    current_Qn += w_Qn
    print("{:25} | Q{:2}.{}".format('post input*weight', current_Qm, current_Qn))
    b_Qm = weight_qformat['fc3_b_0']['m']
    b_Qn = weight_qformat['fc3_b_0']['n']
    print("{:25} | Q{:2}.{}".format('fc3 bias', b_Qm, b_Qn))
    bias_shift = current_Qn - b_Qn
    assert bias_shift>=0, "this is a rare case, pls debug bias_shift"
    print("{:25} | Qn from {} to {}".format('bias shift (left)', b_Qn, current_Qn))

    final_in_qformat = derive_qformat(act_stats['fc3/fc3/Max_0']['mean'])
    _, next_Qm, next_Qn, _, _ = final_in_qformat
    print("{:25} | Q{}.{} -> Q{}.{}".format(
        'fc3_out -> final_input', 
        current_Qm, current_Qn,
        next_Qm, next_Qn))
    out_shift = current_Qn - next_Qn 
    assert out_shift>=0, "this is a rare case, pls debug out_shift"
    print("{:25} | Qn from {} to {}".format('out shift (right)', current_Qn, next_Qn))
    print('-'*100)
    print("fc3 | bias_shift = {} | out_shift = {}".format(bias_shift, out_shift))
    print('_'*100+"\n")

    # final
    #---
    _, current_Qm, current_Qn, current_qmin, _ = final_in_qformat
    print("{:25} | Q{:2}.{} | act_max: {}".format('final input', current_Qm, current_Qn, -1*(current_qmin)))
    w_Qm = weight_qformat['final_fc_0']['m']
    w_Qn = weight_qformat['final_fc_0']['n']
    print("{:25} | Q{:2}.{}".format('final weight', w_Qm, w_Qn))
    current_Qm += w_Qm
    current_Qn += w_Qn
    print("{:25} | Q{:2}.{}".format('post input*weight', current_Qm, current_Qn))
    b_Qm = weight_qformat['Variable_0']['m']
    b_Qn = weight_qformat['Variable_0']['n']
    print("{:25} | Q{:2}.{}".format('final bias', b_Qm, b_Qn))
    bias_shift = current_Qn - b_Qn
    assert bias_shift>=0, "this is a rare case, pls debug bias_shift"
    print("{:25} | Qn from {} to {}".format('bias shift (left)', b_Qn, current_Qn))

    logits_qformat = derive_qformat(act_stats['logits/Max_1_0']['mean'])
    _, next_Qm, next_Qn, _, _ = logits_qformat
    print("{:25} | Q{}.{} -> Q{}.{}".format(
        'final_out -> logits', 
        current_Qm, current_Qn,
        next_Qm, next_Qn))
    out_shift = current_Qn - next_Qn 
    assert out_shift>=0, "this is a rare case, pls debug out_shift"
    print("{:25} | Qn from {} to {}".format('out shift (right)', current_Qn, next_Qn))
    print('-'*100)
    print("final | bias_shift = {} | out_shift = {}".format(bias_shift, out_shift))
    print('_'*100+"\n")

    _, current_Qm, current_Qn, current_qmin, _ = logits_qformat
    print("{:25} | Q{:2}.{} | act_max: {}".format('logits', current_Qm, current_Qn, -1*(current_qmin)))
    print('_'*100+"\n")

def run_quant_inference(wanted_words, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, dct_coefficient_count, 
                           model_architecture, model_size_info, track_minmax):
  """Creates an audio model with the nodes needed for inference.

  Uses the supplied arguments to create a model, and inserts the input and
  output nodes that are needed to use the graph for inference.

  Args:
    wanted_words: Comma-separated list of the words we're trying to recognize.
    sample_rate: How many samples per second are in the input audio files.
    clip_duration_ms: How many samples to analyze for the audio pattern.
    window_size_ms: Time slice duration to estimate frequencies from.
    window_stride_ms: How far apart time slices should be.
    dct_coefficient_count: Number of frequency bands to analyze.
    model_architecture: Name of the kind of model to generate.
    model_size_info: Model dimensions : different lengths for different models
  """
  
  tf.logging.set_verbosity(tf.logging.INFO)
  sess = tf.InteractiveSession()
  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, dct_coefficient_count)

  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings)
  
  label_count = model_settings['label_count']
  fingerprint_size = model_settings['fingerprint_size']

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  handles = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      FLAGS.model_size_info,
      FLAGS.act_max,
      is_training=False, track_minmax=track_minmax)

  if track_minmax and FLAGS.model_architecture=='dnn': 
    
    dt = datetime.now()
    ts_track_minmax_dir = os.path.join(FLAGS.track_minmax_dir,'{:%Y-%m-%d__%H-%M-%S}'.format(dt))
    os.makedirs(ts_track_minmax_dir, exist_ok=False) # will error out if the folder exists because we want a unique event file in the folder

    writer = tf.summary.FileWriter(ts_track_minmax_dir)

    # track minmax only support 
    logits, summary_handle_list = handles
  else:
    logits = handles

  ground_truth_input = tf.placeholder(
      tf.float32, [None, label_count], name='groundtruth_input')

  predicted_indices = tf.argmax(logits, 1)
  expected_indices = tf.argmax(ground_truth_input, 1)
  correct_prediction = tf.equal(predicted_indices, expected_indices)
  confusion_matrix = tf.confusion_matrix(
      expected_indices, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  models.load_variables_from_checkpoint(sess, FLAGS.checkpoint)


  # Quantize weights to 8-bits using (min,max) and write to file
  f = open('weights.h','wb')
  f.close()

  weight_qformat = OrderedDict()

  for v in tf.trainable_variables():
    var_name = str(v.name)
    var_values = sess.run(v)
    min_value = var_values.min()
    max_value = var_values.max()
    int_bits = int(np.ceil(np.log2(max(abs(min_value),abs(max_value)))))
    if int_bits < 0:
        int_bits = 0 # int_bits will be negative when the max(abs()) is lower than 1
    dec_bits = 7-int_bits
    # convert to [-128,128) or int8
    var_values = np.round(var_values*2**dec_bits)
    var_name = var_name.replace('/','_')
    var_name = var_name.replace(':','_')
    with open('weights.h','a') as f:
      f.write('#define '+var_name+' {')
    if(len(var_values.shape)>2): #convolution layer weights
      transposed_wts = np.transpose(var_values,(3,0,1,2))
    else: #fully connected layer weights or biases of any layer
      transposed_wts = np.transpose(var_values)
    with open('weights.h','a') as f:
      transposed_wts.tofile(f,sep=", ",format="%d")
      f.write('}\n')
    # convert back original range but quantized to 8-bits or 256 levels
    var_values = var_values/(2**dec_bits)
    # update the weights in tensorflow graph for quantizing the activations
    var_values = sess.run(tf.assign(v,var_values))
    print(var_name+\
            '; number of wts/bias: '+str(var_values.shape)+\
            '; Q-format: Q{}.{}'.format(int_bits, dec_bits)+\
            '; int bits: '+str(int_bits)+\
            '; frac bits: '+str(dec_bits)+\
            '; float min: {}'.format(min_value)+\
            '; Q min: {}'.format(var_values.min())+\
            '; float max: {}'.format(max_value)+\
            '; Q max: {}'.format(var_values.max()))
    
    weight_qformat[var_name] = {
        'm': int_bits, 
        'n': dec_bits,
    }

  # training set
  set_size = audio_processor.set_size('training')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in range(0, set_size, FLAGS.batch_size):
    training_fingerprints, training_ground_truth = (
        audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                 0.0, 0, 'training', sess))
    output_handle_list = [evaluation_step, confusion_matrix]
    if track_minmax and FLAGS.model_architecture=='dnn':
        output_handle_list += summary_handle_list
    fetches = sess.run(
        output_handle_list,
        feed_dict={
            fingerprint_input: training_fingerprints,
            ground_truth_input: training_ground_truth,
        })
    
    if track_minmax and FLAGS.model_architecture=='dnn':
      training_accuracy, conf_matrix = fetches[:2]
      minmax_summ = fetches[2:]
      for each in minmax_summ:
        writer.add_summary(each, i) # add summary
    else:
      training_accuracy, conf_matrix = fetches

    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (training_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Training accuracy = %.2f%% (N=%d)' %
                  (total_accuracy * 100, set_size))

  # validation set
  set_size = audio_processor.set_size('validation')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in range(0, set_size, FLAGS.batch_size):
    validation_fingerprints, validation_ground_truth = (
        audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                 0.0, 0, 'validation', sess))
    validation_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: validation_fingerprints,
            ground_truth_input: validation_ground_truth,
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (validation_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Validation accuracy = %.2f%% (N=%d)' %
                  (total_accuracy * 100, set_size))
  
  # test set
  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in range(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                           set_size))

  # post processing on tracked statistics - simulate and suggest bias and output shifting
  if track_minmax is True:
    writer.flush()
    writer.close()

    writer_eventfile_list = glob(f'{ts_track_minmax_dir}/events.out.tfevents.*')
    if len(writer_eventfile_list) != 1:
        ValueError("Expect one eventfile generated when track_minmax is enabled, pls debug.")
    else:
        event_file = sorted(writer_eventfile_list)[-1] # Assumption that the last file is the latest generated event file"
        
        output_pth = os.path.join(
            os.path.dirname(event_file), 
            "csv." + os.path.basename(event_file))

        summarized_output_path = os.path.join(
            os.path.dirname(event_file), 
            "summ.csv." + os.path.basename(event_file))
        
        df = eventfile_to_df(event_file)
        df.to_csv(output_pth, index=True)
        summ_df = df.describe()
        summ_df.to_csv(summarized_output_path)
        print("[Info]: {}, {} are generated.".format(output_pth, summarized_output_path))
        
        suggest_bias_and_out_shift(weight_qformat, summ_df)   

def main(_):

  # Create the model, load weights from checkpoint and run on train/val/test
  run_quant_inference(FLAGS.wanted_words, FLAGS.sample_rate,
      FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count,
      FLAGS.model_architecture, FLAGS.model_size_info,
      FLAGS.track_minmax)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--checkpoint',
      type=str,
      default='',
      help='Checkpoint to load the weights from.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='dnn',
      help='What model architecture to use')
  parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128,128,128],
      help='Model dimensions - different for various models')
  parser.add_argument(
      '--act_max',
      type=float,
      nargs="+",
      default=[128,128,128],
      help='activations max')
  parser.add_argument(
      '--track_minmax', action='store_true')
  parser.add_argument(
      '--track_minmax_dir',
      type=str,
      default="./act_stats",
      help='directory that saves the tensorflow event files and min max statistics. default is "./act_stats/')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
