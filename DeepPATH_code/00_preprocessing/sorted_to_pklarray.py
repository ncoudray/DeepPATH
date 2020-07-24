'''
The MIT License (MIT)

Copyright (c) 2019, Nicolas Coudray and Aristotelis Tsirigos (NYU)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated doc
umentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERC
HANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This program converts images to numpay arrays and save them as pickle
Does it for test, train and valid sets
labels are saved in vector (like label=3 out of 5 possible will be coded as 00100)

'''


import os
from PIL import Image
from array import *
import random 
import tensorflow as tf
import pickle as pkl
import numpy as np


tf.app.flags.DEFINE_string('input_dir', '/path_to_sorted_jpg_images/',
                           'Training data directory')
tf.app.flags.DEFINE_string('MaxNbImages', '-1,-1,-1',
                            'Maximum number of images in each class for each set (train,valid,test, in that order, as string, separated by comma only) - Will be taken randomly if >0, otherwise, all images are taken if set to -1 (may help in unbalanced datasets: undersample oneof the datasets)')
tf.app.flags.DEFINE_integer('cropsize', 1000,
                           'if images need to be cropped')
tf.app.flags.DEFINE_boolean('gl', False,
                           'if images need to be converted to Grey-level (1 channel)')

FLAGS = tf.app.flags.FLAGS

def _find_image_files(name, data_dir, vMaxNbImages):
  unique_labels = []
  for item in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, item)):
                unique_labels.append(os.path.join(item))
  unique_labels.sort()
  print("unique_labels:")
  print(unique_labels)
  labels = []
  filenames = []
  texts = []
  # Leave label index 0 empty as a background class.
  label_index = 1
  for text in unique_labels:
    typeIm = name + '*.jpeg'
    jpeg_file_path = os.path.join(data_dir, text, typeIm)
    matching_files = tf.gfile.Glob(jpeg_file_path)
    if len(matching_files) < 1:
      typeIm = name + '*.jpg'
      jpeg_file_path = os.path.join(data_dir, text, typeIm)
      matching_files = tf.gfile.Glob(jpeg_file_path)
    if vMaxNbImages > 0:
      tmp_label = [label_index] * len(matching_files)
      tmp_text = [text] * len(matching_files)
      tmp_filename = matching_files
      shuffled_index = list(range(len(tmp_filename)))
      random.seed(12345)
      random.shuffle(shuffled_index)
      tmp_label = [tmp_label[i] for i in shuffled_index]
      tmp_text = [tmp_text[i] for i in shuffled_index]
      tmp_filename = [tmp_filename[i] for i in shuffled_index]
      # Take FLAGS.MaxNbImages images
      tmp_label = tmp_label[:min(vMaxNbImages, len(tmp_filename))]
      tmp_text = tmp_text[:min(vMaxNbImages, len(tmp_filename))]
      tmp_filename = tmp_filename[:min(vMaxNbImages, len(tmp_filename))]
      #print("length filename:%d  " % len(tmp_filename))
      #print(tmp_label)
      #print(tmp_text)
      #print("%d  " % tmp_filename)
      labels.extend(tmp_label)
      texts.extend(tmp_text)
      filenames.extend(tmp_filename)
      print("length filename (no tmp):%d  " % len(filenames))
    else:
      labels.extend([label_index] * len(matching_files))
      texts.extend([text] * len(matching_files))
      filenames.extend(matching_files)
    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1
  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)
  filenames = [filenames[i] for i in shuffled_index]
  texts = [texts[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]
  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
  for nn in range(len(filenames)):
    print(filenames[nn], texts[nn], labels[nn])
  return filenames, texts, labels



'''
def convert(FileList
  for imagename in FileList:
    data_image = array('B')
    im = Image.open(imagename)
    pixel = im.load()
    width, height = im.size
    for x in range(0,width):
      for y in range(0,height):
        data_image.append(pixel[y,x])
'''

def convert_pickle(filenames, labels, dset):
	ws = FLAGS.cropsize
	gl = FLAGS.gl
	all_img = []
	all_lab = []
	NbImg = len(filenames)
	for kk in range(len(filenames)):
		if gl:
			img = Image.open(filenames[kk]).convert('L')
			imgn = np.asarray(img)
			# imgn = imgn[:ws,:ws]
		else:
			img = Image.open(filenames[kk])
			imgn = np.asarray(img)
			# imgn = imgn[:ws,:ws,:]
		for xI in range(int(imgn.shape[0]/ws)):
			for yI in range(int(imgn.shape[1]/ws)):
				if gl:
					imgn2 = imgn[xI*ws:xI*ws+ws,yI*ws:yI*ws+ws]
				else:
					imgn2 = imgn[xI*ws:xI*ws+ws,yI*ws:yI*ws+ws,:]
				labv = np.zeros(max(labels))
				labv[labels[kk]-1] = 1
				print(imgn2.shape) 
				all_img.append(imgn2)
				all_lab.append(labv)
	all_img = np.array(all_img,dtype='float32')
	all_lab =  np.array(all_lab, dtype='float64')
	fileObject = open(dset + "_" + str(ws) + str(gl) + '_' + str(NbImg) + 'img.pkl', 'wb')
	pkl.dump(all_img, fileObject, protocol=4)
	fileObject.close()
	fileObject = open(dset + "_" + str(ws) + str(gl) + '_' + str(NbImg) + 'label.pkl', 'wb')
	pkl.dump(all_lab, fileObject, protocol=4)
	fileObject.close()
	return all_img, all_lab

def main(unused_argv):
  print(FLAGS.MaxNbImages)
  vMaxNbImages= FLAGS.MaxNbImages.split(',')
  vMaxNbImages = [int(x) for x in vMaxNbImages]
  print("MaxNb or images:")
  print(vMaxNbImages)
  print(vMaxNbImages[0])
  print(vMaxNbImages[2])
  filenames, texts, labels = _find_image_files('train', FLAGS.input_dir, vMaxNbImages[0])
  convert_pickle(filenames, labels, 'train')
  filenames, texts, labels = _find_image_files('test', FLAGS.input_dir, vMaxNbImages[2])
  convert_pickle(filenames, labels, 'test')


if __name__ == '__main__':
  tf.app.run()


