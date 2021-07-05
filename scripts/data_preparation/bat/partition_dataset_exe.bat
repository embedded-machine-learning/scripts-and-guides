echo #====================================#
echo #Prepare dataset for training
echo #====================================#
echo #Partition dataset in a training and validation set

:: Constant Definition
set USERNAME=wendt
set USEREMAIL=alexander.wendt@tuwien.ac.at
set MODELNAME=tf2oda_ssdmobilenetv2_320_320_coco17_D100_pedestrian
set PYTHONENV=tf24
set BASEPATH=.
set SCRIPTPREFIX=..\..\Projekte\21_SoC_EML\scripts-and-guides\scripts

:: Environment preparation
echo Activate environment %PYTHONENV%
call conda activate %PYTHONENV%

::    # Initiate argument parser
::    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
::                                     formatter_class=argparse.RawTextHelpFormatter)
::    parser.add_argument(
::        '-i', '--imageDir',
::        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
::        type=str,
::        default=os.getcwd()
::    )
::    parser.add_argument(
::        '-o', '--outputDir',
::        help='Path to the output folder where the train and test dirs should be created. '
::             'Defaults to the same directory as IMAGEDIR.',
::        type=str,
::        default=None
::    )
::    parser.add_argument(
::        '-d', '--xmlDir',
::        help='Path to the folder where the xml bounding boxes are stored. If not specified, '
::             'the image directory will be used.',
::        type=str,
::        default=None
::    )
::    parser.add_argument(
::        '-r', '--ratio',
::        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
::        default=0.1,
::        type=float)
::    parser.add_argument(
::        '-x', '--xml_copy',
::        help='Set this flag if you want the xml annotation files to be processed and copied over.',
::        action='store_true'
::    )
::    parser.add_argument('-m', '--remove_source',
::                        help='Set this flag to remove the source images',
::                       action='store_true')

python %SCRIPTPREFIX%\partition_dataset.py ^
-i "images" ^
--xmlDir "annotations/xmls" ^
-r 0.10 ^
--remove_source

rem #python 020_partition_dataset.py -i "images" --xmlDir "annotations/xmls" --file_id_dir "annotations" -r 0.10