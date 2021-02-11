import glob
import os
import sys

from tensorflow.core.util import event_pb2
import tensorflow.compat.v2 as tf
from absl import flags
import pandas as pd
import numpy as np

flags.DEFINE_string('eval_file', None, 'Path to evaluation file ' 'file.')
flags.DEFINE_string('checkpoint_dir', None, "Directory with the checkpoints and evaluation results. Read the newest file in the folder.")
flags.DEFINE_string('out_dir', None, 'Path to output')

FLAGS = flags.FLAGS

# This example supposes that the events file contains summaries with a
# summary value tag 'loss'.  These could have been added by calling
# `add_summary()`, passing the output of a scalar summary op created with
# with: `tf.scalar_summary(['loss'], loss_tensor)`.


def main(unused_argv):
    metrics = pd.DataFrame(columns=["step", "value"])
    metrics.index.rename("name", inplace=True)

    #path_to_events_file = FLAGS.eval_file
    #out_path = FLAGS.out_path

    datafile_path = None

    if FLAGS.checkpoint_dir is not None:
        # Get the newest file from eval
        eval_dir = os.path.join(FLAGS.checkpoint_dir, "eval")
        files = glob.glob(eval_dir + "/*")
        datafile_path = max(files, key=os.path.getctime)

        model_name = os.path.basename(os.path.normpath(FLAGS.checkpoint_dir))

    elif FLAGS.eval_file is not None:
        if os.path.isfile(FLAGS.eval_file)==False:
            raise FileNotFoundError("File does not exist: ", FLAGS.eval_file)
        datafile_path = FLAGS.eval_file

        model_name = os.path.basename(os.path.normpath(datafile_path))
    else:
        raise FileNotFoundError("No data source. Either the checkpoint_dir must be set or the eval_file.")

    print("Got model and output file name: ", model_name)

    serialized_examples = tf.data.TFRecordDataset(datafile_path)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for value in event.summary.value:
            t = tf.make_ndarray(value.tensor)

            #print(value.tag, event.step, t, type(t))
            if str(value.tag).find("eval_side_by_side")<0:
                #print("{}={}".format(value.tag, t))
                s=pd.Series(data=[event.step, t], name=value.tag, index=["step", "value"])
                metrics = metrics.append(s)

    print("Collected metrics: \n", metrics)

    out_path = os.path.join(FLAGS.out_dir, model_name + ".csv")
    if os.path.isdir(FLAGS.out_dir)==False:
        os.makedirs(FLAGS.out_dir)
    metrics.to_csv(out_path)

    print("Program End")
    sys.exit(0) #Exit code, else program throws error


if __name__ == '__main__':
  tf.compat.v1.app.run()
