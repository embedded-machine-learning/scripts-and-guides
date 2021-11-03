import argparse
import io
import os
import sys

import tensorflow as tf
import PIL
import cv2

def main(argv):
    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Clean images for Tensorflow.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--image_dir',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        default=os.getcwd()
    )

    args = parser.parse_args()

    clean_images(args.image_dir)

    print("All images in {} processed".format(args.image_dir))


def clean_images(image_dir):
    path_images = image_dir  # './images'
    filenames_src = tf.io.gfile.listdir(path_images)
    for filename_src in filenames_src:
        stem, extension = os.path.splitext(filename_src)
        if (extension.lower() != '.jpg'): continue

        pathname_jpg = '{}/{}'.format(path_images, filename_src)
        with tf.io.gfile.GFile(pathname_jpg, 'rb') as fid:
            encoded_jpg = fid.read(4)
        # png
        if (encoded_jpg[0] == 0x89 and encoded_jpg[1] == 0x50 and encoded_jpg[2] == 0x4e and encoded_jpg[3] == 0x47):
            # copy jpg->png then encode png->jpg
            print('png:{}'.format(filename_src))
            pathname_png = '{}/{}.png'.format(path_images, stem)
            tf.io.gfile.copy(pathname_jpg, pathname_png, True)
            PIL.Image.open(pathname_png).convert('RGB').save(pathname_jpg, "jpeg")
            # gif
        elif (encoded_jpg[0] == 0x47 and encoded_jpg[1] == 0x49 and encoded_jpg[2] == 0x46):
            # copy jpg->gif then encode gif->jpg
            print('gif:{}'.format(filename_src))
            pathname_gif = '{}/{}.gif'.format(path_images, stem)
            tf.io.gfile.copy(pathname_jpg, pathname_gif, True)
            PIL.Image.open(pathname_gif).convert('RGB').save(pathname_jpg, "jpeg")
        elif (filename_src == 'beagle_116.jpg' or filename_src == 'chihuahua_121.jpg'):
            # copy jpg->jpeg then encode jpeg->jpg
            print('jpeg:{}'.format(filename_src))
            pathname_jpeg = '{}/{}.jpeg'.format(path_images, stem)
            tf.io.gfile.copy(pathname_jpg, pathname_jpeg, True)
            PIL.Image.open(pathname_jpeg).convert('RGB').save(pathname_jpg, "jpeg")
        elif (encoded_jpg[0] != 0xff or encoded_jpg[1] != 0xd8 or encoded_jpg[2] != 0xff):
            print('not jpg:{}'.format(filename_src))

        #https://stackoverflow.com/questions/33548956/detect-avoid-premature-end-of-jpeg-in-cv2-python
        with open(pathname_jpg, 'rb') as im:
            im.seek(-2, 2)
            if im.read() == b'\xff\xd9':
                print('Image OK :', pathname_jpg)
            else:
                # fix image
                img = cv2.imread(pathname_jpg)
                cv2.imwrite(pathname_jpg, img)
                print('FIXED corrupted image :', pathname_jpg)



if __name__ == "__main__":
    sys.exit(int(main(sys.argv) or 0))