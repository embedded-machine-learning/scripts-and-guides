import os
import zipfile
import glob
import itertools

from absl import app
from absl import flags

flags.DEFINE_string('items', None, 'List of folders and files to compress into one zip. Folders are separated by space')
flags.DEFINE_string('out', None, 'Output Zip file name')

FLAGS = flags.FLAGS

def zipdir(path, ziph):
    # ziph is zipfile handle
    if os.path.isfile(path):
        ziph.write(path)
        print("Added ", path)
    else:
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(path, '..')))
                print("Added ", file)


def zipit(dir_list, zip_name):
    zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for dir in dir_list:
        zipdir(dir, zipf)
    zipf.close()
    print("Saved zip file to ", zip_name)

#def open_files(path):
#    for filename in glob.glob(path):


def main(argv):
    flags.mark_flag_as_required('items')
    flags.mark_flag_as_required('out')

    folders = FLAGS.items
    folders_list = folders.split(",")
    extened_list = list(itertools.chain(*[glob.glob(str(path).strip()) for path in folders_list]))

    zipit(extened_list, FLAGS.out)



if __name__ == '__main__':
    app.run(main)