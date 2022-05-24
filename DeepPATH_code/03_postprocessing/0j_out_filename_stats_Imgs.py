import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import matplotlib.image as mpimg
import glob
from matplotlib import cm

parser = argparse.ArgumentParser()

parser.add_argument(
      '--files_stats',
      type=str,
      default='out_filename_Stats.txt',
      help="out_filename_Stats.txt"
)
parser.add_argument(
      '--labelFile',
      type=str,
      default='',
      help="File with label names, 1 per line"
)
parser.add_argument(
      '--outputPath',
      type=str,
      default='',
      help="Path to save output files"
)
parser.add_argument(
      '--Mag',
      type=str,
      default='20.0',
      help="Magnification (tile path sub-folder name)"
)
parser.add_argument(
      '--tilesPath',
      type=str,
      default='',
      help="Path to tiled svs"
)
args = parser.parse_args()


imagePath = args.tilesPath
Mag = args.Mag
outputPath = args.outputPath
labelFile = args.labelFile
tiles_stats = args.files_stats

text_file = open(labelFile)
AllLabels = ['Bkg']
AllLabels.extend(text_file.read().split('\n'))
print(AllLabels)
if AllLabels[-1]=='':
	AllLabels = AllLabels[:-1]

slideN = []
TileN = []
SlideTileN = []
Probs = {}
for kk in range(len(AllLabels)):
	Probs[kk] = []
#Probs[0] = []
#Probs[1] = []
#Probs[2] = []
#Probs[3] = []
#Probs[4] = []

TrueLabel = []

# tiles_stats = 'out_filename_Stats.txt'

with open(tiles_stats) as f:
	for line in f:
		line2 = line.replace('[','').replace(']','').split()
		if len(line2)>0:
			tilename = '.'.join(line2[0].split('.')[:-1])
			cTileRootName =  '_'.join(os.path.basename(tilename).split('_')[0:-2])
			ixTile = int(os.path.basename(tilename).split('_')[-2])
			iyTile = int(os.path.basename(tilename).split('_')[-1].split('.')[0])
			lineProb = line.split('[')[1]
			lineProb = lineProb.split(']')[0]
			lineProb = lineProb.split()
			slideN.extend([cTileRootName])
			TileN.extend([str(ixTile) + "_" + str(iyTile)])			
			SlideTileN.extend([tilename])
			for kk in range(len(AllLabels)):
			        Probs[kk].append(float(lineProb[kk]))
			#Probs[0].append(float(lineProb[0]))
			#Probs[1].append(float(lineProb[1]))
			#Probs[2].append(float(lineProb[2]))
			#Probs[3].append(float(lineProb[3]))
			#Probs[4].append(float(lineProb[4]))
			TrueLabel.append(int(line2[-1]))





df = pd. DataFrame({'slide_name': slideN, 'Tile_ID': TileN, 'Tile_name':SlideTileN, 'Prob_0': Probs[0], 'Prob_1': Probs[1], 'Prob_2': Probs[2], 'Label':TrueLabel})
for kk in range(len(AllLabels)):
	df[AllLabels[kk]] = Probs[kk]



df = df.drop_duplicates()

#  df[df['Label']==2].sort_values(by='Prob_1', ascending=False)[1:20]

df.to_csv(os.path.join(outputPath, 'out_filename_Stats_unique.csv'))



# imagePath = '/gpfs/data/abl/deepomics/jourlab/Tiling/299px_full_2021_10_06_NYU/'
# Mag = '20.0'




# RefLabel = 2
# RefLabel_Prob = 'Prob_2'
# for eachLabel in ['Prob_0','Prob_1','Prob_2','Prob_3','Prob_4']:
for RefLabel in range(1, len(AllLabels)):
# for RefLabel_Prob in AllLabels:
	RefLabel_Prob = AllLabels[RefLabel]
	for eachLabel in AllLabels:
		# print(eachLabel)
		df_sorted = df[df['Label']==RefLabel].sort_values(by=eachLabel, ascending=False)
		dfloop = df_sorted.index
		#fig = plt.figure()
		SlideCount = {}
		image_datas = {}
		image_info = {}
		ImgNb = 0
		for row in dfloop:
			slidename = df_sorted['slide_name'][row].split('test_')[-1]
			tileNumber  =  df_sorted['Tile_ID'][row]
			tilename = os.path.join(imagePath,  slidename + '_files', Mag,  tileNumber + '.jpeg')
			if  not os.path.exists(tilename):
				tilename = os.path.join(imagePath, '*',  slidename + '_files', Mag,  tileNumber + '.jpeg')
				tilename = glob.glob(tilename)[0]
			#print(tilename)
			SProb = df_sorted[eachLabel][row]
			TProb = df_sorted[RefLabel_Prob][row]
			if slidename in SlideCount.keys():
				SlideCount[slidename] = SlideCount[slidename] + 1
			else:
				SlideCount[slidename] = 1
			if SlideCount[slidename] > 6:
				continue
			elif  os.path.exists(tilename):
				image_datas[ImgNb] = mpimg.imread(tilename)
				image_info[ImgNb] = {}
				image_info[ImgNb]['name'] = slidename + '_' + tileNumber
				image_info[ImgNb]['Class probability'] = SProb
				image_info[ImgNb]['Real targetclass probability'] = TProb
				ImgNb += 1
			if ImgNb ==32:
				break
		_, axs = plt.subplots(4, 8, figsize=(16, 8))
		plt.tight_layout(pad=1.2)
		axs = axs.flatten()
		#axs.axis('off')
		for index, ax in zip(range(0,ImgNb), axs):
			ax.imshow(image_datas[index])
			ax.title.set_text(image_info[index]['name'] + ':\n%s: %.3f\n%s: %.3f' % (eachLabel, image_info[index]['Class probability'], RefLabel_Prob, image_info[index]['Real targetclass probability']))
			ax.title.set_fontsize(6)
			#ax.axis('off')
			ax.set_xticks([])
			ax.set_yticks([])
			for side in ax.spines.keys():
				ax.spines[side].set_linewidth(image_info[index]['Class probability']*10)
				ax.spines[side].set_color(cm.jet(image_info[index]['Class probability']))
		plt.savefig(os.path.join(outputPath, RefLabel_Prob + '_as_' + eachLabel + '.png'))
		







