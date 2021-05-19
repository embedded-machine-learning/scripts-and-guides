import os
import argparse
import datetime
import csv
from inference_utils import get_info_from_modelname 
'''
	using the utils from scripts here:
		https://github.com/embedded-machine-learning/scripts-and-guides/blob/main/scripts/inference_evaluation/inference_utils.py
'''

parser = argparse.ArgumentParser(description='Reporting the trtexec command')

parser.add_argument('--onnx', default="/media/cdleml/128GB/Users/amozelli/tf_to_onnx/onnx_models/tfkeras_inceptionv3_299x299_imagenet_.onnx",
                    help='Onnx model path for building TRT engine', required=False)

parser.add_argument('--batch', type=int, default=1,
                    help='Batch Size', required=False)
					
parser.add_argument('--shapes', #type=str,
                    default='data:1x3x299x299',
                    help='Input shape in form of "<NAME>:NxCxHxW"', required=False)
					
parser.add_argument('--precision',
                    help='TensorRT precision mode: fp32, fp16, int8, best', required=True)
					
parser.add_argument('--inputs_dir', default="/media/cdleml/128GB/Users/amozelli/onnx_to_trt/data/",
                    help='For actual input data.', required=False)
                    
'''parser.add_argument('--saveEngine', default='trt_engine.engine',
                    help='Where to save the converted TRT-engine.', required=False)'''
											
parser.add_argument('--loadEngine', default='inception_test.engine',
                    help='Loading existing TRT-engine.', required=False)
					
parser.add_argument('--csv', default='./engine_report.csv',
                    help='Location of the .csv file if exists.', required=False)
                    
parser.add_argument('--buildEngine', action='store_true',
                    help='Set this flag if you want to load an existing TRT-engine.', required=False)
					
args, unknown = parser.parse_known_args()
print(args)



# --batch='+str(args.batch)+'
# --shapes='+args.shapes+' 
# --loadInputs='+args.inputs_dir+'

#build and benchmark
if args.buildEngine:
  args.saveEngine = '/home/stephan/Users/amozelli/onnx_to_trt/trt_engine/'+args.onnx.split('/')[-1].split('.')[0]+'.engine'
  path_list = args.saveEngine
  cmd = 'trtexec --onnx='+args.onnx+' --explicitBatch --saveEngine='+args.saveEngine+' > out_file.txt'
  if args.precision=='fp16':
    cmd = 'trtexec --onnx='+args.onnx+' --explicitBatch --saveEngine='+args.saveEngine+' --fp16 > out_file.txt'
    
  if args.precision=='int8':
    cmd = 'trtexec --onnx='+args.onnx+' --explicitBatch --saveEngine='+args.saveEngine+' --int8 > out_file.txt'
    
  if args.precision=='best':
    cmd = 'trtexec --onnx='+args.onnx+' --explicitBatch --saveEngine='+args.saveEngine+' --best > out_file.txt'
  os.system(cmd)

#benchmark only
else:
  path_list = args.loadEngine
  cmd = 'trtexec  --loadEngine='+args.loadEngine+' --shapes='+args.shapes+' --loadInputs='+args.inputs_dir+' > out_file.txt'
  os.system(cmd) 
  


with open('out_file.txt') as f:
    lines = f.readlines() 
    
for line in lines:
  if 'mean' in line:
    av_time = line.split(' ')[-2]
    #print(line)
    #print(av_time)


# writing csv file
now = datetime.datetime.now()
path_list = path_list.split('/')
header = path_list[-1]
name_list = header.split('_')
'''while len(name_list) < 4:
    name_list.append('ND')'''
    
    
    
#getting info from the model name
info = get_info_from_modelname(header)




headers =     ['ID',
                'Date',
                'Model',
                'Model_Short',
                'Framework',
                'Network',
                'Resolution',
                'Dataset',
                'Custom_Parameters',
                'Hardware',
                'Hardware_Optimization',
                'Precision',
                'Batch_Size',
                'Throughput',
                'Mean_Latency(ms)',
                'Latencies(ms)',]
                #'detection_boxes',
                #'detection_scores',
                #'detection_classes']

'''body =        ['TBD',
                now.strftime('%Y-%m-%d'),# %H:%M:%S"),
                "MODEL_TBD",
                "MODEL_SHORT_TBD",
                name_list[0],
                name_list[1],
                name_list[2],
                name_list[3],
                'TBD',
                "NVIDIA_NANO_TBD",
                'trt-engine',
                args.precision,
                args.batch,
                'TBD',
                av_time[0:3],
                "SINGLE_LATENCIES_TBD",]
                #detection_boxes,
                #detection_scores,
                #detection_classes]'''

body =        ['TBD',
                now.strftime('%Y-%m-%d'),# %H:%M:%S"),
                info['model_name'],
                "MODEL_SHORT_TBD",
                info['framework'],
                info['network'],
                info['resolution'],
                info['dataset'],
                'TBD',
                "NVIDIA_NANO_TBD",
                'trt-engine',
                args.precision,
                args.batch,
                'TBD',
                av_time[0:3],
                "SINGLE_LATENCIES_TBD",]
                #detection_boxes,
                #detection_scores,
                #detection_classes]

    
if os.path.exists(args.csv):
    with open(args.csv, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(body)
                  
else:
    with open(args.csv, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(headers)
        writer.writerow(body)  
    
cmd = 'rm out_file.txt'
os.system(cmd)