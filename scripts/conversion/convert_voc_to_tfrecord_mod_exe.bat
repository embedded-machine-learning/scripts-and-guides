python convert_voc_to_tfrecord_mod.py ^
-x "samples/annotations/xml" ^
-i "samples/images" ^
-l "samples/annotations/sw_label_map.pbtxt" ^
-o "samples/prepared-records/train_voc.record" ^
-n 2