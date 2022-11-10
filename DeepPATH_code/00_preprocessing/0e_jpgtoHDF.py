'''
The MIT License (MIT)

Copyright (c) 2021, Anna Yeaton, Nicolas Coudray

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Authors: Nicolas Coudray, Theodoros Sakellaropoulos
    Date created: March/2017

        Objective:
        Starting with tiles images, select images from a given magnification and order them according the the stage of the cancer, or type of cancer, etc...

        Usage:
                SourceFolder is the folder where all the svs images have been tiled . :
                It encloses:
                  * 1 subfolder per image, (the name of the subfolder being "imagename_files")
                  * each contains 14-17 subfolders which name is the magnification of the tiles and contains the tiles
                It should not enclose any other folder
                The output folder from which the script is run should be empty

'''



import h5py
import glob
import os
import numpy as np
import cv2
from mpi4py import MPI
from itertools import chain
import argparse
import random

# srun -p cpu_short -c 11 -n 11 --mem-per-cpu=4G -t 00-02:00:00  --pty bash

# module unload gcc
# module load openmpi
# module load python/cpu/3.6.5

# mpiexec -n 10 python parallel_hdf5.py


parser = argparse.ArgumentParser(description="Convert jpeg images sorted is subfolders (1 per class) to hdf5 format.")

parser.add_argument("--input_path", type=str, default='',
                        help="input path (parent directory)")
parser.add_argument("--output", type=str, default='path/output.d5',
                        help="path and name of output")
parser.add_argument("--chunks", type=int, default=40,
                        help="must equal the number of cores")
parser.add_argument("--sub_chunks", type=int, default=20,
                        help="number of sub-chunks")
parser.add_argument("--wSize", type=int, default=224,
                        help="output window size")
parser.add_argument("--mode", type=int, default=1,
                        help="0 - path=tiled images in original folder; 1 - path=tiled images sorted in subfolders as labels; 2 - path=sorted images path")
parser.add_argument("--subset", type=str, default='train',
                        help="in mode 2 only, will only take tiles within a certain subset (train, test or valid; or combined) - in mode 1, it will be appened to the tags name within h5")
parser.add_argument("--startS", type=int, default=0,
                        help="First image in the image list to take")
parser.add_argument("--stepS", type=int, default=1,
                        help="set to >1 if sample is needed. Every stepS image will be taken starting with image startS")
parser.add_argument("--maxIm", type=int, default=-1,
                        help="maximum number of images to take. All if set to -1")
parser.add_argument("--mag", type=float, default=2.016,
                        help="magnification to use (mode 0 and 1 only)")
parser.add_argument("--label", type=str, default='',
                        help="label to use (mode 0 only); either a string or a filepath with labels")
parser.add_argument("--slideID", type=int, default='23',
                        help="number of characters to use for the slide ID")
parser.add_argument("--sampleID", type=int, default='14',
                        help="number of characters to use for the sample (or patient) ID")





args = parser.parse_args()

chunks = args.chunks
sub_chunks = args.sub_chunks
top_level = args.input_path

# chunks = 80
# sub_chunks = 20
# get length of data
# top_level = "/gpfs/data/abl/deepomics/osmanlab/melanoma/immunotherapy_project_scans/Processed_paul/P321_from301_TrainAll/"
# /gpfs/home/ay1392/SPORE_R01/Results/Anna/pathgan_tiles/tiles_20x"


WIDTH = args.wSize
HEIGHT = args.wSize

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

# get length of all images to add to hdf5
Images = []
patterns1 = [f.name for f in os.scandir(top_level) if f.is_dir()]
patterns1.sort()
patterns = patterns1
# print(patterns)
if args.mode == 1:
    for pattern in patterns:
        slides = [f.name for f in os.scandir(os.path.join(top_level, pattern)) if f.is_dir()]
        # slides = [f.name for f in os.scandir(os.path.join(top_level, pattern)) if f.is_file()]
        print(pattern)
        for slide in slides:
            images = glob.glob(os.path.join(top_level, pattern, slide, str(args.mag) + "/*.jpeg"))
            Images.append(images)
elif args.mode == 0:
    if os.path.isfile(args.label):
      patterns1 = []
      with open(args.label, "rU") as f:
            jdata = {}
            nameLengthL = 1000000
            nameLengthT = 0
            for line in f:
                tmp_PID = line.split()[0]
                nameLengthT = max(nameLengthT, len(tmp_PID))
                nameLengthL = min(nameLengthL, len(tmp_PID))
                jdata[tmp_PID] = line.split()[1]
                if line.split()[1] not in  patterns1:
                    patterns1.append(line.split()[1])
      print(range(nameLengthT,nameLengthL-1,-1))
    else:
      patterns1 = [args.label]
    # print(jdata)
    slides = [f.name for f in os.scandir(os.path.join(top_level)) if f.is_dir()]
    # slides = [f.name for f in os.scandir(os.path.join(top_level)) if f.is_file()]
    # print(slides)
    for slide in slides:
      images = glob.glob(os.path.join(top_level, slide, str(args.mag) + "/*.jpeg"))
      Images.append(images)
elif args.mode ==2:
    for pattern in patterns:
        if args.subset=='combined':
            images = glob.glob(os.path.join(top_level, pattern, "*.jpeg"))
        else:
            images = glob.glob(os.path.join(top_level, pattern, args.subset + "*.jpeg"))
        Images.append(images)
        print(pattern)
    #slides = [f.name for f in os.scandir(os.path.join(top_level, pattern)) if f.is_dir()]
    #print(pattern)
    #for slide in slides:
    #    images = glob.glob(os.path.join(top_level, pattern, slide, "*.jpeg"))
    #    Images.append(images)
# print("Nb of images:" + str(len(Images)))

# length of all images
ImageList1 = list(chain.from_iterable(Images))

print("aaa1 - " + str(len(ImageList1)))
# print("aaa2 - " + str(len(ImageList)))
# if args.maxTiles < len(ImageList):
if args.startS > len(ImageList1):
    args.startS = 0
    print("startS is larger than the total number of images")
if args.startS < 0:
    args.startS = 0
if args.stepS < 1:
    args.stepS = 1
Indx = [x for x in range(args.startS, len(ImageList1), args.stepS)]
print("aaa3 - " + str(len(Indx)))
# print(Indx)
# random.shuffle(Indx)
# Indx2 = Indx[0:args.maxTiles]
# Indx2.sort(reverse=False)
ImageList = [ImageList1[x] for x in Indx]
print("aaa3 - " + str(len(ImageList)))
if args.maxIm > 0:
    ImageList = ImageList[0:args.maxIm]
print("aaa3 - " + str(len(ImageList)))
# ImageList = Images2


patterns1.append('unknown')
# create hdf5 dataset
f = h5py.File(args.output, 'w', driver='mpio', comm=MPI.COMM_WORLD)
dset = f.create_dataset(args.subset+'_img', (len(ImageList), WIDTH, HEIGHT, 3), dtype='uint8')
dset2 = f.create_dataset(args.subset+'_patterns', (len(ImageList), ),
                         dtype = 'S37')
dset3 = f.create_dataset(args.subset+'_slides', (len(ImageList), ),
                         dtype = 'S23')
dset4 = f.create_dataset(args.subset+'_tiles', (len(ImageList), ),
                         dtype = 'S16')
dset5 = f.create_dataset(args.subset+'_labels', (len(ImageList), ))
dset6 = f.create_dataset(args.subset+'_hist_subtype', (len(ImageList), ),
                         dtype = 'S37')
dset7 = f.create_dataset(args.subset+'_samples', (len(ImageList), ),
                         dtype = 'S23')


# size of the chunks
chunk_size, remainder = divmod(len(ImageList), chunks)
# if remainder>0:
# chunk_size = chunk_size + 1
# remainder = len(ImageList) - (chunk_size * (chunks -1))



print("total images:" + str(len(ImageList)) + ", " + str(chunk_size) + " chunks")

# get start and end indices of the hdf5

## does this over write one space??
if rank == (chunks - 1):
    start = int(rank * chunk_size)
    # end = int(start + remainder)
    end = int(start + remainder + chunk_size)
    print("rank: " + str(rank) + " start: " + str(start) + " end: " + str(end))
else:
    start = int(rank * chunk_size)
    end = int(start + chunk_size)
    print("rank: " + str(rank) + " start: " + str(start) + " end: " + str(end))

# load in data in chunks within this process
ranges = end - start
print("ranges " + str(ranges))
sub_chunk_size, sub_chunk_remainder = divmod(ranges, sub_chunks)
# sub_chunk_size = sub_chunk_size + 1
# sub_chunk_remainder = ranges - (sub_chunk_size * (sub_chunks -1))


for j in range(0,sub_chunks):
   # get start and end indices of the hdf5
   if j == (sub_chunks - 1):
       sub_start = int(start  + sub_chunk_size * j)
       # sub_end = int(sub_start + sub_chunk_remainder)
       sub_end = int(sub_start + sub_chunk_remainder + sub_chunk_size)
       print("rank: " + str(rank) + " chunk: " + str(j) + " start: " + str(sub_start) + " end: " + str(sub_end))
   else:
       sub_start = int(start + sub_chunk_size * j)
       sub_end = int(sub_start + sub_chunk_size)
       print("rank: " + str(rank) + " chunk: " + str(j) + " start: " + str(sub_start) + " end: " + str(sub_end))

   loadedImages = []
   loadedPatterns = []
   loadedSlides = []
   loadedTiles = []
   loadedLabels = []
   loadedSamples = []
   for i, img in enumerate(ImageList[sub_start:sub_end]):
       #print(img)
       #print(os.stat(img).st_size)
       image = cv2.imread(img)
       if args.mode == 0:
           slide = img.split("/")[-3]
           tile = img.split("/")[-1]
           if os.path.isfile(args.label):
               for nameLength in range(nameLengthT,nameLengthL-1,-1):
                   try:
                       pattern = jdata[slide[0:nameLength]]
                       break
                   except:
                       pattern = 'unknown'
           else:
               pattern = args.label
       if args.mode == 1:
           pattern = img.split("/")[-4]
           slide = img.split("/")[-3]
           tile = img.split("/")[-1]
       if args.mode == 2:
           pattern = img.split("/")[-2]
           slide_tmp = img.split("/")[-1]
           slide = slide_tmp.split("_")[1]
           tile = "_".join(slide_tmp.split("_")[-2:])
       side = slide + "_" + pattern
       #pattern = img.split("/")[9]
       #slide = img.split("/")[10]
       #print("shape:")
       #print(image.shape[0], WIDTH, image.shape[1], HEIGHT)
       try:
           print(img, image.shape[0], pattern)
       except:
           print("error 1:" + img)
           # print(ImageList[sub_start:sub_end])
       if max(image.shape[0]/float(WIDTH), image.shape[1]/float(HEIGHT)) > 1:  
           nfac = 1./max(image.shape[0]/float(WIDTH), image.shape[1]/float(HEIGHT))
           try:
                image = cv2.resize(image, (0,0), fx=nfac, fy=nfac)
           except:
                print("error 2:" + img + " " + str(nfac) + " " + str(image.shape[0]) + " " + str(image.shape[1]))
       if image.shape[0] > WIDTH:
           image = image[:WIDTH,:,:]
       if image.shape[1] > HEIGHT:
           image =image[:,:HEIGHT,:]
       image = np.uint8(image)
       sample = slide[:args.sampleID]
       slide = slide[:args.slideID]
       #print("shape:")
       #print(image.shape[0], WIDTH, image.shape[1], HEIGHT)
       if image.shape[0] == WIDTH and image.shape[1] == HEIGHT:
           loadedImages.append(image)
           loadedPatterns.append(pattern)
           loadedSlides.append(slide)
           loadedTiles.append(tile)
           loadedSamples.append(sample)
           try:
              loadedLabels.append(patterns1.index(pattern))
           except:
              loadedLabels.append(-1)
       else:
           #print("ELSE")
           delta_h = HEIGHT - image.shape[0]
           delta_w = WIDTH - image.shape[1]
           top, remainder_vert = divmod(delta_h, 2)
           top = top + remainder_vert
           bottom, remainder_vert = divmod(delta_h, 2)
           left, remainder_horz = divmod(delta_w, 2)
           left = left + remainder_horz
           right, remainder_horz = divmod(delta_w, 2)
           color = [255, 255, 255]
           #print("cv2:" + str(top) + "_" + str(botton) + "_" + str(right))
           #print(cv2.BORDER_CONSTANT)
           image = cv2.copyMakeBorder(src=image, top=top,
                                      bottom=bottom,
                                      left=left,
                                      right=right, borderType=cv2.BORDER_CONSTANT, value=color)
           loadedImages.append(image)
           loadedPatterns.append(pattern)
           loadedSlides.append(slide)
           loadedTiles.append(tile)
           loadedLabels.append(patterns1.index(pattern))
           loadedSamples.append(sample)

   print("***********" + str(rank) + "; " + str(sub_chunks) + "; "  + str(j) + "; "  + str(sub_chunk_size) + " to " + str(len(loadedImages)))
   # print(sub_chunks)
   # print(j)
   # print(len(loadedImages))
   # print(len(ImageList[sub_start:sub_end]))
   # print(loadedImages)
   image_stack = np.stack(loadedImages, axis=0)
   pattern_stack = np.stack(loadedPatterns, axis=0).astype('S37')
   slide_stack = np.stack(loadedSlides, axis=0).astype('S23')
   #tile_stack =  np.stack(loadedTiles, axis=0).astype('S9')
   tile_stack =  np.stack(loadedTiles, axis=0).astype('S16')
   labels_stack = np.stack(loadedLabels, axis=0)
   sample_stack = np.stack(loadedSamples, axis=0).astype('S23')


   dset[sub_start:sub_end, ...] = image_stack
   dset2[sub_start:sub_end, ...] = pattern_stack
   dset3[sub_start:sub_end, ...] = slide_stack
   dset4[sub_start:sub_end, ...] = tile_stack
   dset5[sub_start:sub_end, ...] = labels_stack
   dset6[sub_start:sub_end, ...] = pattern_stack
   dset7[sub_start:sub_end, ...] = sample_stack

# print("closing:")
# len(dset)
# len(dset2)
# len(dset3)
# len(dset4)
# len(dset5)
f.close()
print("job " + str(j) + " finished properly")
