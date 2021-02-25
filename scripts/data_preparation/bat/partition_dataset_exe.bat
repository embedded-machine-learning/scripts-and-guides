echo #====================================#
echo #Prepare dataset for training
echo #====================================#
echo #Partition dataset in a training and validation set

python partition_dataset.py ^
-i "images" ^
--xmlDir "annotations/xmls" ^
--file_id_dir "annotations" ^
-r 0.10 ^
--remove_source

rem #python 020_partition_dataset.py -i "images" --xmlDir "annotations/xmls" --file_id_dir "annotations" -r 0.10