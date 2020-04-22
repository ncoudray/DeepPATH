
from PIL import Image, ImageDraw, ImageCms
import numpy as np
import sys
import getopt
from skimage import color, io

def RGB_to_lab(tile):
    # srgb_p = ImageCms.createProfile("sRGB")
    # lab_p  = ImageCms.createProfile("LAB")
    # rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    # Lab = ImageCms.applyTransform(tile, rgb2lab)
    # Lab = np.array(Lab)
    # Lab = Lab.astype('float')
    # Lab[:,:,0] = Lab[:,:,0] / 2.55
    # Lab[:,:,1] = Lab[:,:,1] - 128
    # Lab[:,:,2] = Lab[:,:,2] - 128
    Lab = color.rgb2lab(tile)
    return Lab



def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile="])
    except getopt.GetoptError:
        print("test.py -i \<inputfile\>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i inputfile')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    try:
        tile = Image.open(inputfile)
        Lab = RGB_to_lab(tile)
        print("LAB values:")
        for i in range(3):
            print("Channel " + str(i) + " has mean of " + str(np.mean(Lab[:,:,i])) + " and std of " + str(np.std(Lab[:,:,i])))
    except Exception as e:
                print(e)

if __name__== "__main__":
    print(sys.argv[1:])
    main(sys.argv[1:])
