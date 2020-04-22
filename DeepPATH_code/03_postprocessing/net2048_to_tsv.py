"""
The MIT License (MIT)

Copyright (c) 2017, Nicolas Coudray and Aristotelis Tsirigos (NYU)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


	Date created: September/2017

	Get the net2048 files (last layer of inception. 1 file per tile) and aggregate the results into a TSV file if a given condition is met

	usage:
		net2048_to_tsv.py --net2048_dir '/full_path_to/net2048/' --output_dir 'output/' --tile_filename_Stats 'test/test_200000k/out_filename_Stats.txt'

"""


import argparse
import os.path
import glob

FLAGS = None

def main():
	dico = {}
	if os.path.isfile(FLAGS.tile_filename_Stats):
		Filter = True
		with open(FLAGS.tile_filename_Stats) as f:
			for line in f:
				(key, val) = line.split(".dat")
				dico[key] = val
	else:
		Filter = False
		print(FLAGS.tile_filename_Stats + " NOT FOUND")
		quit()

	with open(os.path.join(FLAGS.output_dir,FLAGS.outbasename+"lastlayer_net2048.tsv"), "w") as mydatafile:
		with open(os.path.join(FLAGS.output_dir,FLAGS.outbasename+"lastlayer_trueLabels.tsv"),"w") as TL_file, open(os.path.join(FLAGS.output_dir,FLAGS.outbasename+"lastlayer_predictedProbs.tsv"),"w") as PP_file, open(os.path.join(FLAGS.output_dir,FLAGS.outbasename+"lastlayer_predictedLabels.tsv"),"w") as PL_file, open(os.path.join(FLAGS.output_dir,FLAGS.outbasename+"tile_list.txt"),"w") as jpg_file:
			for tile in glob.glob(os.path.join(FLAGS.net2048_dir, "*.net2048")):
				tileID = ".".join(os.path.basename(tile).split(".")[:-1])
				ValidEntry = True
				if tileID in dico:
					basename = "_".join(tileID.split("_")[:-2])
					ClassData = dico[tileID].replace('[','').replace(']','').split()
					PredictedLabels = [float(x) for x in ClassData[2:-3]]
					nMax = max(PredictedLabels)
					classMax = PredictedLabels.index(max(PredictedLabels))
					TrueLabel_Indx = int(ClassData[-1])
					content = open(tile).readlines()
					mydatafile.write("\n")
					mydatafile.write(" ".join(content).replace("\n",""))
					setLabels = []                                         
					for kk in range(len(PredictedLabels)):
						setLabels.append('0')
					setLabels[classMax] = '1'
					TrueLabels = []
					for kk in range(len(PredictedLabels)):					
						TrueLabels.append('0')
					TrueLabels[TrueLabel_Indx-1] = '1'

					print(tileID)
					print(TrueLabels)
					TL_file.write("\n" + " ".join(TrueLabels))
					PP_file.write("\n" + " ".join(ClassData[2:-3]))
					PL_file.write("\n" + " ".join(setLabels))
					jpg_file.write("\n" + tileID + "\t" + basename)
				else:
					print("tile ID not found in out_filename_Stats: " + tileID)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--net2048_dir',
		type=str,
		default='',
		help='directory where the net2048 files are located (last layer of CNN).'
	)
	parser.add_argument(
		'--output_dir',
		type=str,
		default='',
		help='Output files.tsv.'
	)
	parser.add_argument(
		'--tile_filename_Stats',
		type=str,
		default='',
		help='out_filename_Stats.txt'
	)
	parser.add_argument(
		'--outbasename',
		type=str,
		default='',
		help='basename for the output files'
	)
	FLAGS, unparsed = parser.parse_known_args()
	main()

