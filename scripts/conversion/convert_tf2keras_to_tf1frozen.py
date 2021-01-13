import os
import sys

import tensorflow as tf
#from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
#import numpy as np
from tensorflow.keras.models import load_model
#from tensorflow.python.tools import optimize_for_inference_lib
from absl import flags, app
#from subprocess import call

FLAGS = flags.FLAGS
#flags.DEFINE_string('myflag', 'Some default string', 'The value of myflag.')
flags.DEFINE_string('input', '', 'input h5 model path with weights')
flags.DEFINE_string('output', '', 'output frozen model path')

def main(argv):
    #print("Architecture: ", FLAGS.architecture)
    #architecture = FLAGS.architecture
    print("Output path ", FLAGS.output)
    output_path = FLAGS.output
    print("Model input ", FLAGS.input)
    model_path = FLAGS.input

    frozen_out_path = ''# name of the .pb file
    #frozen_graph_filename = architecture+"/frozen_"+model

    frozen_graph_filename = output_path
    #model = load_model(architecture+"/"+model+".h5")
    model = load_model(model_path)

    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))# Get frozen ConcreteFunction

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    print(frozen_func.inputs)
    print(frozen_func.outputs)

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 60)
    print("Frozen model layers: ")
    for layer in layers:
            print(layer)
            print("-" * 60)
            print("Frozen model inputs: ")
            print(frozen_func.inputs)
            print("Frozen model outputs: ")
            print(frozen_func.outputs)# Save frozen graph to disk
            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                logdir=frozen_out_path,
                name=f"{frozen_graph_filename}",
                as_text=False)# Save its text representation
            """ 
            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                logdir=frozen_out_path,
                name=f"{frozen_graph_filename}.pbtxt",
                as_text=True)
            """
    #rc = call("python -m tensorflow.python.tools.optimize_for_inference --input ./"+architecture+"/frozen_graph.pb --output ./"+architecture+"/optimized_graph.pb --frozen_graph=True --input_names=x --output_names=Identity")

    print("Network converted to frozen model.")
    sys.exit(0)

if __name__ == '__main__':
    app.run(main)
