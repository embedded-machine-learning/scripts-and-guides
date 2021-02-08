rem python "C:/Program Files (x86)/IntelSWTools/openvino_2021/deployment_tools/model_optimizer/mo_tf.py" --input_model "C:/Projekte/21_SoC_EML/Tensorflow_Object_Detection_tf2/workspace/star_wars_recognition_task_model/exported-models/ssd_mobilenet_v2_300x300_coco17_starwars/saved_model/saved_model.pb" --transformations_config "C:/Program Files (x86)/IntelSWTools/openvino_2021/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json" --tensorflow_object_detection_api_pipeline_config "C:/Projekte/21_SoC_EML/Tensorflow_Object_Detection_tf2/workspace/star_wars_recognition_task_model/exported-models/ssd_mobilenet_v2_300x300_coco17_starwars/pipeline.config"
rem python "C:/Program Files (x86)/IntelSWTools/openvino_2021/deployment_tools/model_optimizer/mo_tf.py" --input_model "C:/Projekte/21_SoC_EML/Tensorflow_Object_Detection_tf2/workspace/star_wars_recognition_task_model/exported-models/ssd_mobilenet_v2_300x300_coco17_starwars/saved_model/saved_model.pb" --transformations_config "C:/Program Files (x86)/IntelSWTools/openvino_2021/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json" --tensorflow_object_detection_api_pipeline_config "C:/Projekte/21_SoC_EML/Tensorflow_Object_Detection_tf2/workspace/star_wars_recognition_task_model/exported-models/ssd_mobilenet_v2_300x300_coco17_starwars/pipeline.config"
rem python "C:/Program Files (x86)/IntelSWTools/openvino_2021/deployment_tools/model_optimizer/mo_tf.py" ^
rem --input_model "C:/Projekte/21_SoC_EML/Tensorflow_Object_Detection_tf2/workspace/star_wars_recognition_task_model/exported-models/ssd_mobilenet_v2_300x300_coco17_starwars/saved_model/saved_model.pb" ^
rem --transformations_config "C:/Program Files (x86)/IntelSWTools/openvino_2021/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json" ^
rem --tensorflow_object_detection_api_pipeline_config "C:/Projekte/21_SoC_EML/Tensorflow_Object_Detection_tf2/workspace/star_wars_recognition_task_model/exported-models/ssd_mobilenet_v2_300x300_coco17_starwars/pipeline.config"

python "C:/Program Files (x86)/IntelSWTools/openvino_2021/deployment_tools/model_optimizer/mo_tf.py" ^
--saved_model_dir "C:/Projekte/21_SoC_EML/Tensorflow_Object_Detection_tf2/workspace/star_wars_recognition_task_model/exported-models/ssd_mobilenet_v2_300x300_coco17_starwars/saved_model" ^
--tensorflow_object_detection_api_pipeline_config "C:/Projekte/21_SoC_EML/Tensorflow_Object_Detection_tf2/workspace/star_wars_recognition_task_model/exported-models/ssd_mobilenet_v2_300x300_coco17_starwars/pipeline.config" ^
--log_level=INFO

rem --input_shape "[1 300 300 3]" ^

rem --transformations_config "C:/Program Files (x86)/IntelSWTools/openvino_2021/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json" ^
rem --input_model "C:/Projekte/21_SoC_EML/Tensorflow_Object_Detection_tf2/workspace/star_wars_recognition_task_model/exported-models/ssd_mobilenet_v2_300x300_coco17_starwars/saved_model/saved_model.pb" ^

rem python "C:/Program Files (x86)/IntelSWTools/openvino_2021/deployment_tools/model_optimizer/mo_tf.py" ^
rem --input_model="C:/Projekte/21_SoC_EML/Tensorflow_Object_Detection_tf2/workspace/star_wars_recognition_task_model/pre-trained-models/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb" ^
rem --transformations_config "C:/Program Files (x86)/IntelSWTools/openvino_2021/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json" ^
rem --tensorflow_object_detection_api_pipeline_config "C:/Projekte/21_SoC_EML/Tensorflow_Object_Detection_tf2/workspace/star_wars_recognition_task_model/pre-trained-models/ssd_inception_v2_coco_2018_01_28/pipeline.config" ^
rem --reverse_input_channels