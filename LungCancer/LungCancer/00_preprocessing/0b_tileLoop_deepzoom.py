'''
    File name: 0b_tileLoop_deepzoom.py
    Author: Nicolas Coudray
    Date created: March/2017
    Date last modified: 2/25/2017
    Python Version: 2.7 (native on the cluster

	Objective:
	Tile svs images

	Usage:
	python 0b_tileLoop_deepzoom.py <svs images path> <tile_size> <overlap> <number of processes> <number of threads> <Max Percentage of Background>'
		modify tile size and input directory in the code itself

	Be careful:
	Overload of the node? 

	Initial tests:
	tested on Test_20_tiled/Test2 and Test5  using imgExample = "/ifs/home/coudrn01/NN/Lung/Test_20imgs/*/*svs"

'''

import subprocess
from glob import glob
from multiprocessing import Process, JoinableQueue
import time
import os
import sys

def ImgWorker(queue):
	print("ImgWorker started")
	while True:
		cmd = queue.get()			
		if cmd is None:
			queue.task_done()
			break
		print("Execute: %s" % (cmd))
		subprocess.Popen(cmd, shell=True).wait()
		queue.task_done()


if __name__ == '__main__':
	# Initialization
	#imgExample = "/ifs/home/coudrn01/NN/Lung/Test_20imgs/*/*svs"
	#tile_size = 1024  
	# imgExample = "/ifs/home/coudrn01/NN/Lung/RawImages/*/*svs"
	# tile_size = 512
	# max_number_processes = 10
	# NbrCPU = 4

	if len(sys.argv) != 7:
		print('Usage: %prog <svs images path> <tile_size> <overlap> <number of processes> <number of threads> <Max Percentage of Background>')
		print("Example: python 0b_tileLoopdeepzoom,py '/ifs/home/coudrn01/NN/Lung/RawImages/*/*svs' 512 0 10 4 20")
		sys.exit()

	imgExample = sys.argv[1]
	tile_size = int(sys.argv[2])
	overlap = int(sys.argv[3])
	max_number_processes = int(sys.argv[4])
	NbrCPU = int(sys.argv[5])
	MaxBkg = float(sys.argv[6])

	# get  images from the data/ file.
	files = glob(imgExample)  
	files
	len(files)
	# print(files)
	print("***********************")

	dz_queue = JoinableQueue()
	procs = []
	for i in range(max_number_processes):
		p = Process(target = ImgWorker, args = (dz_queue,))
		p.deamon = True
		p.start()
		procs.append(p)

	for imgNb in range(len(files)):
		filename = files[imgNb]
		print(filename)
		basename = os.path.splitext(os.path.basename(filename))[0]
		print(basename)
		if os.path.isdir("%s_files" % (basename)):
			print("EXISTS")
		else:
			print("Not Found")
		cmd = "python ../../nc_deepzoom_tile.py -e %d -j %d -f jpeg -s %d -Bkg %f %s " %(overlap, NbrCPU, tile_size, MaxBkg, filename) 
		dz_queue.put(cmd)

	dz_queue.join()
	for i in range(max_number_processes):
		dz_queue.put( None )

	print("End")


