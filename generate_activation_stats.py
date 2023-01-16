import argparse
import tensorflow as tf
from collections import OrderedDict
import pandas as pd
import os

def parse_event_to_dict(e):
    # this routine is must comply to our tracking design, 
    # do align accordingly whenever tracking mechanism is changed
    d = OrderedDict()
    d["step"] = e.step
    d["tag"] = e.summary.value[0].tag
    d["value"] = e.summary.value[0].simple_value
    return d

def eventfile_to_df(event_filepath):
    step_dict = OrderedDict() # each step has only one entry in dataframe
    for e in tf.compat.v1.train.summary_iterator(event_filepath):
        if len(e.summary.value) > 0:
            d = parse_event_to_dict(e)

            if d['step'] not in step_dict:
                step_dict[d['step']] = dict()
            
            if d['tag'] in step_dict[d['step']]:
                raise ValueError("{} key pair has already defined in step {}, potential duplicates and logical error".format(d['tag'], d['step']))
            else:
                step_dict[d['step']][d['tag']] = d['value']

    return pd.DataFrame.from_dict(step_dict).T

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--event_file',
      type=str,
      default=None,
      help="""\
        input file, the event file generated from quant_test.py with --track_minmax
      """)
    parser.add_argument(
      '--output_csv',
      type=str,
      default=None,
      help="""\
        output csv
      """)
    
    FLAGS, unparsed = parser.parse_known_args()
    
    if FLAGS.event_file is None:
        ValueError("No event_file provided.")
    
    if FLAGS.output_csv is None:
        output_pth = os.path.join(
            os.path.dirname(FLAGS.event_file), 
            "csv." + os.path.basename(FLAGS.event_file))

        summarized_output_path = os.path.join(
            os.path.dirname(FLAGS.event_file), 
            "summ.csv." + os.path.basename(FLAGS.event_file))
        
        df = eventfile_to_df(FLAGS.event_file)
        df.to_csv(output_pth, index=True)
        df.describe().to_csv(summarized_output_path)

        print("[Info]: {}, {} are generated.".format(output_pth, summarized_output_path))