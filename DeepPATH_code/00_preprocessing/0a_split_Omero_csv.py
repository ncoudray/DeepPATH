import csv
import argparse

# Example to remove spaces for filenames, run firt:
# for f in *csv; do mv -- "$f" "${f// /_}"; done


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input',
    type=str,
    help="cvs file", default="")

args = parser.parse_args()

print(args)
csvdir = args.input
# csvdir = "brain_20210111_pathROIs_test1.csv"


f = open(csvdir, 'r')
reader = csv.reader(f)
headers = next(reader, None)
xmlcontent = {}
for ncol in headers:
	xmlcontent[ncol] = []


for nrow in reader:
	for h, r in zip(headers, nrow):
		xmlcontent[h].append(r)

f.close()

f = open(csvdir, 'r')
Lines = f.readlines()
f.close()


ndict = {}
uniqueIm = set(xmlcontent['image_name'])
for kk in uniqueIm:
	kk2 = '['.join(kk.split('[')[:-1])
	if kk2 not in ndict.keys():
		ndict[kk2] = Lines[0]
# print(uniqueIm)
# print(ndict)	

for kk in range(len(xmlcontent['image_name'])):
	kk2 = xmlcontent['image_name'][kk]
	kk3 = '['.join(kk2.split('[')[:-1])
	ndict[kk3] = ndict[kk3] + Lines[kk+1]


for kk in uniqueIm:
	kk2 = '['.join(kk.split('[')[:-1])
	fname = ' '.join(kk.split()[:-1])
	fname = '.'.join(fname.split('.')[:-1])
	f = open( fname + '.csv','w')
	f.write(ndict[kk2])
	f.close()




