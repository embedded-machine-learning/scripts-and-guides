python convert_coco_to_tfrecord.py ^
--logtostderr --image_dir="samples/images" ^
--annotations_file="samples/annotations/coco_train_annotations.json" ^
--output_path="samples/prepared-records/train_coco.record" ^
--number_shards=2