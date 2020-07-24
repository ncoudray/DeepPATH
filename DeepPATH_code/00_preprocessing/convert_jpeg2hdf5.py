import h5py
import glob
import argparse
import os
import numpy as np
import cv2


def get_parser():
    # parse parameters
    parser = argparse.ArgumentParser(description="Unsupervised feature learning.")

    parser.add_argument("--input", type=str, default='',
                        help="input paths (comma separated) where jped are sorted per class (on subfolder per class)")
    parser.add_argument("--output", type=str, default='hdf5_brain',
                        help="path and basename of output")
    parser.add_argument("--labels", type=str, default='labels.txt',
                        help="label text file (1 label per line - must correspond to subfolder names in input directories)")
    parser.add_argument("--subsets", type=str, default='train,valid,test',
                        help="comma separated list of subsets to convert)")
    parser.add_argument("--wSize", type=int, default=224,
                        help="output tile size (will be cropped if smaller than current size)")


    return parser.parse_args()
'''
    Will generate a h5 file for each subset with 2 fields:
    subset_images
    subset_images 
 
    for the labels, they will numbered (from 0) and correspond to the line number in the labels input file

    and an additional text file listing the images names for each entry

'''


def main(args):
  WIDTH = args.wSize
  HEIGHT = args.wSize
  # read list of possible labels (each image should be in a directory with the directory name being the label)
  # image names should start with train, test or valid
  file1 = open(args.labels, 'r') 
  LabelList = file1.read().splitlines() 
  file1.close()

  # read all directories where such images could be found
  indir_list = args.input.split(',')


  hdf5_path = args.output
  All_imgs = {}
  All_labels = {}
  Count = {}
  # subsets = ['train','valid','test']
  subsets = args.subsets.split(',')
  print("List images")
  for subset in range(len(subsets)):
  # with h5py.File(hdf5_path + "_" + subsets[subset] + ".h5",'w') as hdf5_file: 
    # list all images in each subset and their associated label 
    #for subset in range(len(subsets)):
    All_imgs[subsets[subset]] = []
    All_labels[subsets[subset]] = []
    Count[subsets[subset]] = 0
    for cur_dir in indir_list:
      for nLab in range(len(LabelList)):
        imgs = glob.glob(os.path.join(cur_dir, LabelList[nLab], subsets[subset] + '*jpeg'))
        All_imgs[subsets[subset]].extend(imgs)
        All_labels[subsets[subset]].extend(np.full((len(imgs),), nLab))
        Count[subsets[subset]] = Count[subsets[subset]] + len(imgs)
    # save images in h5 format
  print("create h5")
  for subset in range(len(subsets)):
    img_list_file = open(hdf5_path + "_" + subsets[subset] + "_imgList.txt",'w')
    with h5py.File(hdf5_path + "_" + subsets[subset] + ".h5",'w') as hdf5_file:
      print(subsets[subset])
      img_db_shape = (Count[subsets[subset]], WIDTH, HEIGHT, 3)
      hdf5_file.create_dataset(name=subsets[subset] + '_img',maxshape=img_db_shape, dtype=np.uint8, shape=img_db_shape)
      labels_db_shape = (Count[subsets[subset]],)
      hdf5_file.create_dataset(name=subsets[subset]+'_labels', maxshape=labels_db_shape, dtype=np.float32, shape=labels_db_shape)
      # hdf5_file.create_dataset(name=subsets[subset]+'_name', maxshape=labels_db_shape, dtype=np.int, shape=labels_db_shape)
      IndX = 0
      for img, lab in zip(All_imgs[subsets[subset]], All_labels[subsets[subset]]):
        image = cv2.imread(img)
        IndX
        if image.shape[0] >= WIDTH and image.shape[1] >= HEIGHT:
          hdf5_file[subsets[subset] + '_img'][IndX, ...] = image[:WIDTH,:HEIGHT,:]
          hdf5_file[subsets[subset] + '_labels'][IndX, ...] = lab
          # hdf5_file[subsets[subset] + '_name'][IndX, ...] = img
          img_list_file.write(str(IndX) + "\t" + img + "\n")
          IndX += 1
        if IndX % 1000 == 0:
          print(str(IndX)+ " imgs done")
      print(str(IndX)+ " imgs done")
      hdf5_file[subsets[subset] + '_img'].resize((IndX, WIDTH, HEIGHT, 3))
      hdf5_file[subsets[subset] + '_labels'].resize((IndX, ))
      #hdf5_file[subsets[subset] + '_name'].resize((IndX, ))
    img_list_file.close()

'''
    for subset in range(len(subsets)):
      img_db_shape = (Count[subsets[subset]], WIDTH, HEIGHT, 3)
      # img_storage[subset] =
      hdf5_file.create_dataset(name=subsets[subset] + '_img',shape=img_db_shape, dtype=np.uint8)
      labels_db_shape = (Count[subsets[subset]],)
      # label_storage[subset] =  
      hdf5_file.create_dataset(name=subsets[subset]+'_labels', shape=labels_db_shape, dtype=np.float32)
      IndX = 0
      for img, lab in zip(All_imgs[subsets[subset]], All_labels[subsets[subset]]):
        image = cv2.imread(img)
        if image.shape[0] >= WIDTH and image.shape[1] >= HEIGHT:
          hdf5_file[subsets[subset] + '_img'][Indx, ...] = image[:WIDTH,:HEIGHT,:]
          hdf5_file[subsets[subset] + '_labels'][Indx, ...] = lab
          IndX += 1
'''

if __name__ == '__main__':

    # generate parser / parse parameters
    args = get_parser()

    # run experiment
    main(args)
                      
