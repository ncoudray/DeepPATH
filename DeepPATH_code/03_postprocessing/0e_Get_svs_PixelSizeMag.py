import argparse
import os.path
from openslide import open_slide, ImageSlide
import sys

FLAGS = None

def main():
        slide = open_slide(FLAGS.WSI)
        # calculate the best window size before rescaling to reach desired final pizelsize
        try:
                Objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
                OrgPixelSizeX = float(slide.properties['openslide.mpp-x'])
                OrgPixelSizeY = float(slide.properties['openslide.mpp-y'])
        except:
                try:
                        for nfields in slide.properties['tiff.ImageDescription'].split('|'):
                                if 'AppMag' in nfields:
                                        Objective = float(nfields.split(' = ')[1])
                                if 'MPP' in nfields:
                                        OrgPixelSizeX = float(nfields.split(' = ')[1])
                                        OrgPixelSizeY = OrgPixelSizeX
                except:
                        Objective = 1.
                        OrgPixelSizeX = 0
                        print("Error: No information found in the header")
        print(OrgPixelSizeX, Objective)
        if FLAGS.PxsMag == "PixelSize":
                sys.exit(OrgPixelSizeX) 
        elif FLAGS.PxsMag == "Mag":
                sys.exit(Objective)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--WSI',
      type=str,
      default='',
      help='original whole slide image.'
  )
  parser.add_argument(
      '--PxsMag',
      type=str,
      default='Mag',
      help='set to Mag or PixelSize depending on what output you want.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  print(FLAGS)
  main()

