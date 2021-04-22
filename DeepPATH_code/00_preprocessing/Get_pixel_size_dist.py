""" 
The MIT License (MIT)

Copyright (c) 2017, Nicolas Coudray and Aristotelis Tsirigos (NYU)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import openslide
from openslide import open_slide, ImageSlide
import os
from glob import glob
from openslide.deepzoom import DeepZoomGenerator
import sys

#slidepath = "/gpfs/data/abl/deepomics/lung_cancer/TCGA_diagnostic_ffpe/Raw_Diagnostic/*/*svs"
# Mag = 20

def main(argv):
	slidepath = sys.argv[1]
	print(slidepath)
	Mag = float(sys.argv[2])
	print(Mag)

	#try:
	#	opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	#except getopt.GetoptError:
	#	print 'Get_pixel_size_dist.py -i <inputfile> -m <magnification>'
	#	sys.exit(2)


	files = glob(slidepath)
	files = sorted(files)

	with open('pixelsizes_at_' + str(Mag) + 'x.txt','w+') as f:
		for imgNb in range(len(files)):
			filename = files[imgNb]
			basenameJPG = os.path.splitext(os.path.basename(filename))[0].split('.')[0]
			_slide = open_slide(filename)
			dz = DeepZoomGenerator(_slide)
			try:
				OrgPixelSizeX = float(_slide.properties['openslide.mpp-x'])
				OrgPixelSizeY = float(_slide.properties['openslide.mpp-y'])
			except:
				f.write('%s\t%f\tunknown\tunknown\n' %(basenameJPG,ThisMag) )
				continue
			Factors = _slide.level_downsamples
			Objective = float(_slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
			Available = tuple(Objective / x for x in Factors)
			for level in range(dz.level_count-1,-1,-1):
				ThisMag = Available[0]/pow(2,dz.level_count-(level+1))
				if ThisMag != Mag:
					continue
				else:
					print("%s\t%f\t%f\t%f\n" %(basenameJPG,ThisMag, OrgPixelSizeX* pow(2,dz.level_count-(level+1)), OrgPixelSizeY* pow(2,dz.level_count-(level+1))))
					f.write('%s\t%f\t%f\t%f\n' %(basenameJPG,ThisMag, OrgPixelSizeX* pow(2,dz.level_count-(level+1)), OrgPixelSizeY* pow(2,dz.level_count-(level+1))))
if __name__== "__main__":
	    main(sys.argv[1:])






