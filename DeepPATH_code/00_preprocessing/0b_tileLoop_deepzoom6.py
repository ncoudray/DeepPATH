'''
    File name: 0b_tileLoop_deepzoom.py
    Date created: March/2017

	Source:
	Tiling code inspired from
	https://github.com/openslide/openslide-python/blob/master/examples/deepzoom/deepzoom_tile.py
	which is Copyright (c) 2010-2015 Carnegie Mellon University
	The code has been extensively modified 

	Objective:
	Tile svs, jpg or dcm images with the possibility of rejecting some tiles based based on xml or jpg masks

	Be careful:
	Overload of the node - may have memory issue if node is shared with other jobs.
'''

from __future__ import print_function
import json
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
from optparse import OptionParser
import re
import shutil
from unicodedata import normalize
import numpy as np
import scipy.misc
import subprocess
from glob import glob
from multiprocessing import Process, JoinableQueue
import time
import os
import sys
import dicom
# from scipy.misc import imsave
from imageio import imwrite as imsave
# from scipy.misc import imread
from imageio import imread
# from scipy.misc import imresize
import cv2
from xml.dom import minidom
from PIL import Image, ImageDraw, ImageCms
from skimage import color, io
Image.MAX_IMAGE_PIXELS = None


VIEWER_SLIDE_NAME = 'slide'


class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,quality, _Bkg, _ROIpc):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._slide = None
        self._Bkg = _Bkg
        self._ROIpc = _ROIpc

    def RGB_to_lab(self, tile):
        # srgb_p = ImageCms.createProfile("sRGB")
        # lab_p  = ImageCms.createProfile("LAB")
        # rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
        # Lab = ImageCms.applyTransform(tile, rgb2lab)
        # Lab = np.array(Lab)
        # Lab = Lab.astype('float')
        # Lab[:,:,0] = Lab[:,:,0] / 2.55
        # Lab[:,:,1] = Lab[:,:,1] - 128
        # Lab[:,:,2] = Lab[:,:,2] - 128
        # print("RGB to Lab")
        Lab = color.rgb2lab(tile)
        return Lab

    def Lab_to_RGB(self,Lab):
        # srgb_p = ImageCms.createProfile("sRGB")
        # lab_p  = ImageCms.createProfile("LAB")
        # lab2rgb = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "LAB", "RGB")
        # Lab[:,:,0] = Lab[:,:,0] * 2.55
        # Lab[:,:,1] = Lab[:,:,1] + 128
        # Lab[:,:,2] = Lab[:,:,2] + 128
        # newtile = ImageCms.applyTransform(Lab, lab2rgb)
        # print("Lab to RGB")
        newtile = (color.lab2rgb(Lab) * 255).astype(np.uint8)
        return newtile


    def normalize_tile(self, tile, NormVec):
        Lab = self.RGB_to_lab(tile)
        TileMean = [0,0,0]
        TileStd = [1,1,1]
        newMean = NormVec[0:3] 
        newStd = NormVec[3:6]
        for i in range(3):
            TileMean[i] = np.mean(Lab[:,:,i])
            TileStd[i] = np.std(Lab[:,:,i])
            # print("mean/std chanel " + str(i) + ": " + str(TileMean[i]) + " / " + str(TileStd[i]))
            tmp = ((Lab[:,:,i] - TileMean[i]) * (newStd[i] / TileStd[i])) + newMean[i]
            if i == 0:
                tmp[tmp<0] = 0 
                tmp[tmp>100] = 100 
                Lab[:,:,i] = tmp
            else:
                tmp[tmp<-128] = 128 
                tmp[tmp>127] = 127 
                Lab[:,:,i] = tmp
        tile = self.Lab_to_RGB(Lab)
        return tile

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz(None, self._tile_size)
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            #associated, level, address, outfile = data
            associated, level, address, outfile, format, outfile_bw, PercentMasked, SaveMasks, TileMask, Normalize, isrescale, resize_ratio, Adj_WindowSize = data
            if last_associated != associated:
                dz = self._get_dz(associated, self._tile_size)
                last_associated = associated
            # try:
            dz = self._get_dz(associated, Adj_WindowSize)
            # except Exception as e: 
            #     print("****** ERROR: ")
            #    print(e)
            #    print(associated)
            #    break
            #try:
            if True:
                try:
                    tile = dz.get_tile(level, address)
                    # A single tile is being read
                    #check the percentage of the image with "information". Should be above 50%
                    gray = tile.convert('L')
                    bw = gray.point(lambda x: 0 if x<220 else 1, 'F')
                    arr = np.array(np.asarray(bw))
                    avgBkg = np.average(bw)
                    bw = gray.point(lambda x: 0 if x<220 else 1, '1')
                    # check if the image is mostly background
                    #print("res: " + outfile + " is " + str(avgBkg))
                    if avgBkg <= (self._Bkg / 100.0):
                        # print("PercentMasked: %.6f, %.6f" % (PercentMasked, self._ROIpc / 100.0) )
                        # if an Aperio selection was made, check if is within the selected region
                        if PercentMasked >= (self._ROIpc / 100.0):

                            if Normalize != '':
                                # print("normalize " + str(outfile))
                                # arrtile = np.array(tile)
                                tile = Image.fromarray(self.normalize_tile(tile, Normalize).astype('uint8'),'RGB')

                            if (isrescale) and (resize_ratio != 1):
                                # tile.save(outfile + '_orig.jpeg', quality=self._quality)
                                print(tile.width, resize_ratio, tile.height, int(tile.width / resize_ratio), int(tile.height / resize_ratio))
                                tile = tile.resize(  (min( self._tile_size, int(tile.width / resize_ratio) ), min( self._tile_size, int(tile.height / resize_ratio))))
                                # tile = cv2.resize(tile, (0, 0), fx = 1/resize_ratio, fy = 1/resize_ratio)
                                # if tile.shape[0] > self._tile_size:
                                #    tile = tile[:self._tile_size,:,:]
                                #    print("*** Warning: image " + str(outfile) + " had to be cropped from " + str(tile.shape[0]))
                                #if tile.shape[1] > self._tile_size:
                                #    tile = tile[:,:self._tile_size,:]
                                #    print("*** Warning: image " + str(outfile) + " had to be cropped from " + str(tile.shape[1]))
   
 
                            tile.save(outfile, quality=self._quality)
                            if bool(SaveMasks)==True:
                                height = TileMask.shape[0]
                                width = TileMask.shape[1]
                                TileMaskO = np.zeros((height,width,3), 'uint8')
                                maxVal = float(TileMask.max())
                                TileMaskO[...,0] = (TileMask[:,:].astype(float)  / maxVal * 255.0).astype(int)
                                TileMaskO[...,1] = (TileMask[:,:].astype(float)  / maxVal * 255.0).astype(int)
                                TileMaskO[...,2] = (TileMask[:,:].astype(float)  / maxVal * 255.0).astype(int)
                                TileMaskO = numpy.array(Image.fromarray(TileMaskO).resize(arr.shape[0], arr.shape[1],3))
                                # TileMaskO = imresize(TileMaskO, (arr.shape[0], arr.shape[1],3))
                                TileMaskO[TileMaskO<10] = 0
                                TileMaskO[TileMaskO>=10] = 255
                                imsave(outfile_bw,TileMaskO) #(outfile_bw, quality=self._quality)

                        #print("%s good: %f" %(outfile, avgBkg))
                    #elif level>5:
                    #    tile.save(outfile, quality=self._quality)
                            #print("%s empty: %f" %(outfile, avgBkg))
                    self._queue.task_done()
                except Exception as e:
                    # print(level, address)
                    print("image %s failed at dz.get_tile for level %f" % (self._slidepath, level))
                    # e = sys.exc_info()[0]
                    print(e)
                    self._queue.task_done()

    def _get_dz(self, associated=None, wsize=299):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, wsize, self._overlap, limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, format, associated, queue, slide, basenameJPG, xmlfile, mask_type, xmlLabel, ROIpc, ImgExtension, SaveMasks, Mag, normalize, Fieldxml, pixelsize, pixelsizerange, Best_level, resize_ratio, Adj_WindowSize):
        self._dz = dz
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._slide = slide
        self._xmlfile = xmlfile
        self._mask_type = mask_type
        self._xmlLabel = xmlLabel
        self._ROIpc = ROIpc
        self._ImgExtension = ImgExtension
        self._SaveMasks = SaveMasks
        self._Mag = Mag
        self._normalize = normalize
        self._Fieldxml = Fieldxml
        self._pixelsize = pixelsize
        self._pixelsizerange = pixelsizerange
        self._Best_level = Best_level
        self._resize_ratio = resize_ratio
        self._Adj_WindowSize = Adj_WindowSize

    def run(self):
        self._write_tiles()
        self._write_dzi()

    def _write_tiles(self):
            ########################################3
            # nc_added
        #level = self._dz.level_count-1
        Magnification = 20
        tol = 2
        #get slide dimensions, zoom levels, and objective information
        Factors = self._slide.level_downsamples
        try:
            Objective = float(self._slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            # print(self._basename + " - Obj information found")
        except:
            print(self._basename + " - No Obj information found")
            print(self._ImgExtension)
            if ("jpg" in self._ImgExtension) | ("dcm" in self._ImgExtension) | ("tif" in self._ImgExtension):
                #Objective = self._ROIpc
                Objective = 1.
                Magnification = Objective
                print("input is jpg - will be tiled as such with %f" % Objective)
            else:
                return
        #calculate magnifications
        Available = tuple(Objective / x for x in Factors)
        #find highest magnification greater than or equal to 'Desired'
        Mismatch = tuple(x-Magnification for x in Available)
        AbsMismatch = tuple(abs(x) for x in Mismatch)
        if len(AbsMismatch) < 1:
          print(self._basename + " - Objective field empty!")
          return
        '''
        if(min(AbsMismatch) <= tol):
            Level = int(AbsMismatch.index(min(AbsMismatch)))
            Factor = 1
        else: #pick next highest level, downsample
            Level = int(max([i for (i, val) in enumerate(Mismatch) if val > 0]))
            Factor = Magnification / Available[Level]
        # end added
        '''
        xml_valid = False
        # a dir was provided for xml files

        '''
        ImgID = os.path.basename(self._basename)
        Nbr_of_masks = 0
        if self._xmlfile != '':
            xmldir = os.path.join(self._xmlfile, ImgID + '.xml')
            print("xml:")
            print(xmldir)
            if os.path.isfile(xmldir):
                xml_labels, xml_valid = self.xml_read_labels(xmldir)
                Nbr_of_masks = len(xml_labels)
            else:
                print("No xml file found for slide %s.svs (expected: %s). Directory or xml file does not exist" %  (ImgID, xmldir) )
                return
        else:
            Nbr_of_masks = 1
        '''

        if True:
            #if self._xmlfile != '' && :
            # print(self._xmlfile, self._ImgExtension)
            ImgID = os.path.basename(self._basename)
            # xmldir = os.path.join(self._xmlfile, ImgID + '.xml')
            if os.path.isfile(os.path.join(self._xmlfile, ImgID + '.xml')):
               # If path exists, Aperio assumed
               xmldir = os.path.join(self._xmlfile, ImgID + '.xml')
               AnnotationMode = 'Aperio'
            elif os.path.isfile(os.path.join(self._xmlfile, ImgID + '.json')):               
               # QuPath assumed
               xmldir = os.path.join(self._xmlfile, ImgID + '.json')
               AnnotationMode = 'QuPath'

            # print("xml:")
            # print(xmldir)
            if (self._xmlfile != '') & (self._ImgExtension != 'jpg') & (self._ImgExtension != 'dcm'):
                # print("read xml file...")
                mask, xml_valid, Img_Fact = self.xml_read(xmldir, self._xmlLabel, self._Fieldxml, AnnotationMode)
                if xml_valid == False:
                    print("Error: xml %s file cannot be read properly - please check format" % xmldir)
                    return
            elif (self._xmlfile != '')  & (self._ImgExtension == 'dcm'):
                # print("check mask for dcm")
                mask, xml_valid, Img_Fact = self.jpg_mask_read(xmldir)
                # mask <-- read mask 
                #  Img_Fact <-- 1
                # xml_valid <-- True if mask file exists.
                if xml_valid == False:
                    print("Error: xml %s file cannot be read properly - please check format" % xmldir)
                    return

            # print("current directory: %s" % self._basename)

            #return
            #print(self._dz.level_count)

            if self._Mag <= 0:
                if self._pixelsize > 0:
                    level_range = [level for level in range(self._dz.level_count-1,-1,-1)]
                    print(self._slide.properties)
                    try:
                        OrgPixelSizeX = float(self._slide.properties['openslide.mpp-x'])
                        OrgPixelSizeY = float(self._slide.properties['openslide.mpp-y'])
                    except:
                       print("Error: no pixelsize found in the header of %s" % self._basename)
                       DesiredLevel = -1 
                       return
                    AllPixelSizeDiffX = [(abs(OrgPixelSizeX * pow(2,self._dz.level_count-(level+1)) - self._pixelsize)) for level in range(self._dz.level_count-1,-1,-1)]
                    AllPixelSizeDiffY = [(abs(OrgPixelSizeY * pow(2,self._dz.level_count-(level+1)) - self._pixelsize)) for level in range(self._dz.level_count-1,-1,-1)]	
                    IndxX = AllPixelSizeDiffX.index(min(AllPixelSizeDiffX))
                    IndxY = AllPixelSizeDiffY.index(min(AllPixelSizeDiffY))
                    levelX = AllPixelSizeDiffX[IndxX]
                    levelY = AllPixelSizeDiffY[IndxY]
                    print("**info levelX:" + str(levelX) + "; self._pixelsizerange" + str(self._pixelsizerange) + "; test: " + str((levelX > self._pixelsizerange) and (self._pixelsizerange >= 0)))
                    if IndxX != IndxY:
                        print("Error: X and Y pixel sizes are too different for %s"  % self._basename)
                        return
                    if (levelX > self._pixelsizerange) and (self._pixelsizerange >= 0):
                        print("Error: no pixelsize within the desired range for %s"  % self._basename)
                        return
                    if (levelY > self._pixelsizerange) and (self._pixelsizerange >= 0):
                        print("Error: no pixelsize within the desired range for %s"  % self._basename)
                        return
                    if self._pixelsizerange < 0:
                        level_range = [level for level in range(self._dz.level_count-1,-1,-1)]
                        IndxX = self._Best_level
                    DesiredLevel = level_range[IndxX]
                    print('**info: OrgPixelSizeX:' + str(OrgPixelSizeX) +'; DesiredLevel:' + str(DesiredLevel))
                    print(AllPixelSizeDiffX)
                    print(level_range)
                    if not os.path.exists(('/'.join(self._basename.split('/')[:-1]))):
                        os.makedirs(('/'.join(self._basename.split('/')[:-1])))
                    with open(os.path.join( ('/'.join(self._basename.split('/')[:-1])) , 'pixelsizes.txt')  , 'a') as file_out:
                        file_out.write(self._basenameJPG + "\t" + str(OrgPixelSizeX*pow(2,IndxX)) + "\t" + str(OrgPixelSizeX*pow(2,IndxX) * self._resize_ratio) + "\n")
 


            for level in range(self._dz.level_count-1,-1,-1):
                ThisMag = Available[0]/pow(2,self._dz.level_count-(level+1))
                if self._Mag > 0:
                    if ThisMag != self._Mag:
                        continue
                elif self._pixelsize > 0:
                    if level != DesiredLevel:
                        continue
                    else:
                        tiledir_pixel = os.path.join("%s_files" % self._basename, str(self._pixelsize))
                        
                ########################################
                #tiledir = os.path.join("%s_files" % self._basename, str(level))
                tiledir = os.path.join("%s_files" % self._basename, str(ThisMag))
                if not os.path.exists(tiledir):
                    os.makedirs(tiledir)
                    if self._pixelsize >  0:
                        os.symlink(str(ThisMag), tiledir_pixel, target_is_directory=True)
                cols, rows = self._dz.level_tiles[level]
                if xml_valid:
                    # print("xml valid")
                    '''# If xml file is used, check for each tile what are their corresponding coordinate in the base image
                    IndX_orig, IndY_orig = self._dz.level_tiles[-1]
                    CurrentLevel_ReductionFactor = (Img_Fact * float(self._dz.level_dimensions[-1][0]) / float(self._dz.level_dimensions[level][0]))
                    startIndX_current_level_conv = [int(i * CurrentLevel_ReductionFactor) for i in range(cols)]
                    print("***********")
                    endIndX_current_level_conv = [int(i * CurrentLevel_ReductionFactor) for i in range(cols)]
                    endIndX_current_level_conv.append(self._dz.level_dimensions[level][0])
                    endIndX_current_level_conv.pop(0)
    
                    startIndY_current_level_conv = [int(i * CurrentLevel_ReductionFactor) for i in range(rows)]
                    #endIndX_current_level_conv = [i * CurrentLevel_ReductionFactor - 1 for i in range(rows)]
                    endIndY_current_level_conv = [int(i * CurrentLevel_ReductionFactor) for i in range(rows)]
                    endIndY_current_level_conv.append(self._dz.level_dimensions[level][1])
                    endIndY_current_level_conv.pop(0)
                    '''
                    #startIndY_current_level_conv = []
                    #endIndY_current_level_conv = []
                    #startIndX_current_level_conv = []
                    #endIndX_current_level_conv = []

                    #for row in range(rows):
                    #    for col in range(cols):
                    #        Dlocation, Dlevel, Dsize = self._dz.get_tile_coordinates(level,(col, row))
                    #        Ddimension = self._dz.get_tile_dimensions(level,(col, row))
                    #        startIndY_current_level_conv.append(int((Dlocation[1]) / Img_Fact))
                    #        endIndY_current_level_conv.append(int((Dlocation[1] + Ddimension[1]) / Img_Fact))
                    #        startIndX_current_level_conv.append(int((Dlocation[0]) / Img_Fact))
                    #        endIndX_current_level_conv.append(int((Dlocation[0] + Ddimension[0]) / Img_Fact))
                            # print(Dlocation, Ddimension, int((Dlocation[1]) / Img_Fact), int((Dlocation[1] + Ddimension[1]) / Img_Fact), int((Dlocation[0]) / Img_Fact), int((Dlocation[0] + Ddimension[0]) / Img_Fact))
                for row in range(rows):
                    for col in range(cols):
                        InsertBaseName = False
                        if InsertBaseName:
                          tilename = os.path.join(tiledir, '%s_%d_%d.%s' % (
                                          self._basenameJPG, col, row, self._format))
                          tilename_bw = os.path.join(tiledir, '%s_%d_%d_mask.%s' % (
                                          self._basenameJPG, col, row, self._format))
                        else:
                          tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                          col, row, self._format))
                          tilename_bw = os.path.join(tiledir, '%d_%d_mask.%s' % (
                                          col, row, self._format))
                        if xml_valid:
                            # compute percentage of tile in mask
                            # print(row, col)
                            # print(startIndX_current_level_conv[col])
                            # print(endIndX_current_level_conv[col])
                            # print(startIndY_current_level_conv[row])
                            # print(endIndY_current_level_conv[row])
                            # print(mask.shape)
                            # print(mask[startIndX_current_level_conv[col]:endIndX_current_level_conv[col], startIndY_current_level_conv[row]:endIndY_current_level_conv[row]])
                            # TileMask = mask[startIndY_current_level_conv[row]:endIndY_current_level_conv[row], startIndX_current_level_conv[col]:endIndX_current_level_conv[col]]
                            # PercentMasked = mask[startIndY_current_level_conv[row]:endIndY_current_level_conv[row], startIndX_current_level_conv[col]:endIndX_current_level_conv[col]].mean() 
                            # print(startIndY_current_level_conv[row], endIndY_current_level_conv[row], startIndX_current_level_conv[col], endIndX_current_level_conv[col])

                            Dlocation, Dlevel, Dsize = self._dz.get_tile_coordinates(level,(col, row))
                            Ddimension = tuple([pow(2,(self._dz.level_count - 1 - level)) * x for x in self._dz.get_tile_dimensions(level,(col, row))])
                            startIndY_current_level_conv = (int((Dlocation[1]) / Img_Fact))
                            endIndY_current_level_conv = (int((Dlocation[1] + Ddimension[1]) / Img_Fact))
                            startIndX_current_level_conv = (int((Dlocation[0]) / Img_Fact))
                            endIndX_current_level_conv = (int((Dlocation[0] + Ddimension[0]) / Img_Fact))
                            # print(Ddimension, Dlocation, Dlevel, Dsize, self._dz.level_count , level, col, row)

                            #startIndY_current_level_conv = (int((Dlocation[1]) / Img_Fact))
                            #endIndY_current_level_conv = (int((Dlocation[1] + Ddimension[1]) / Img_Fact))
                            #startIndX_current_level_conv = (int((Dlocation[0]) / Img_Fact))
                            #endIndX_current_level_conv = (int((Dlocation[0] + Ddimension[0]) / Img_Fact))
                            TileMask = mask[startIndY_current_level_conv:endIndY_current_level_conv, startIndX_current_level_conv:endIndX_current_level_conv]
                            PercentMasked = mask[startIndY_current_level_conv:endIndY_current_level_conv, startIndX_current_level_conv:endIndX_current_level_conv].mean() 

                            # print(Ddimension, startIndY_current_level_conv, endIndY_current_level_conv, startIndX_current_level_conv, endIndX_current_level_conv)


                            if self._mask_type == 0:
                                # keep ROI outside of the mask
                                PercentMasked = 1.0 - PercentMasked
                                # print("Invert Mask percentage")

                            # if PercentMasked > 0:
                            #     print("PercentMasked_p %.3f" % (PercentMasked))
                            # else:
                            #     print("PercentMasked_0 %.3f" % (PercentMasked))

 
                        else:
                            PercentMasked = 1.0
                            TileMask = []

                        if not os.path.exists(tilename):
                            if self._Best_level == -1:
                                self._queue.put((self._associated, level, (col, row),
                                            tilename, self._format, tilename_bw, PercentMasked, self._SaveMasks, TileMask, self._normalize, False, self._resize_ratio, self._Adj_WindowSize))
                            else:
                                self._queue.put((self._associated, level, (col, row),
                                            tilename, self._format, tilename_bw, PercentMasked, self._SaveMasks, TileMask, self._normalize, True, self._resize_ratio, self._Adj_WindowSize))
                        self._tile_done()

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    self._associated or 'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)

    def _write_dzi(self):
        with open('%s.dzi' % self._basename, 'w') as fh:
            fh.write(self.get_dzi())

    def get_dzi(self):
        return self._dz.get_dzi(self._format)


    def jpg_mask_read(self, xmldir):
        # Original size of the image
        ImgMaxSizeX_orig = float(self._dz.level_dimensions[-1][0])
        ImgMaxSizeY_orig = float(self._dz.level_dimensions[-1][1])
        # Number of centers at the highest resolution
        cols, rows = self._dz.level_tiles[-1]
        # Img_Fact = int(ImgMaxSizeX_orig / 1.0 / cols)
        Img_Fact = 1
        try:
            # xmldir: change extension from xml to *jpg   
            xmldir = xmldir[:-4] + "mask.jpg"
            # xmlcontent = read xmldir image
            xmlcontent = imread(xmldir)
            xmlcontent = xmlcontent - np.min(xmlcontent)
            mask = xmlcontent / np.max(xmlcontent)
            # we want image between 0 and 1
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xmldir)")
            return [], xml_valid, 1.0

        return mask, xml_valid, Img_Fact


    def xml_read(self, xmldir, Attribute_Name, Fieldxml, AnnotationMode):
        # Original size of the image
        ImgMaxSizeX_orig = float(self._dz.level_dimensions[-1][0])
        ImgMaxSizeY_orig = float(self._dz.level_dimensions[-1][1])
        # Number of centers at the highest resolution
        cols, rows = self._dz.level_tiles[-1]

        NewFact = max(ImgMaxSizeX_orig, ImgMaxSizeY_orig) / min(max(ImgMaxSizeX_orig, ImgMaxSizeY_orig),15000.0)
        Img_Fact = float(ImgMaxSizeX_orig) / 5.0 / float(cols)
       

        if AnnotationMode == 'Aperio':
          try:
              xmlcontent = minidom.parse(xmldir)
              xml_valid = True
          except:
              xml_valid = False
              print("error with minidom.parse(xmldir)")
              return [], xml_valid, 1.0

          xy = {}
          xy_neg = {}
          NbRg = 0
          labelIDs = xmlcontent.getElementsByTagName('Annotation')
          # print("%d labels" % len(labelIDs) )
          for labelID in labelIDs:
              if (Attribute_Name==[]) | (Attribute_Name==''):
                      isLabelOK = True
              else:
                  try:
                      labeltag = labelID.getElementsByTagName('Attribute')[0]
                      if (Attribute_Name==labeltag.attributes[Fieldxml].value):
                      # if (Attribute_Name==labeltag.attributes['Value'].value):
                      # if (Attribute_Name==labeltag.attributes['Name'].value):
                          isLabelOK = True
                      else:
                          isLabelOK = False
                  except:
                      isLabelOK = False
              if Attribute_Name == "non_selected_regions":
                  isLabelOK = True

              #print("label ID, tag:")
              #print(labelID, Attribute_Name, labeltag.attributes['Name'].value)
              #if Attribute_Name==labeltag.attributes['Name'].value:
              if isLabelOK:
                  regionlist = labelID.getElementsByTagName('Region')
                  for region in regionlist:
                      vertices = region.getElementsByTagName('Vertex')
                      NbRg += 1
                      regionID = region.attributes['Id'].value + str(NbRg)
                      NegativeROA = region.attributes['NegativeROA'].value
                      # print("%d vertices" % len(vertices))
                      if len(vertices) > 0:
                          #print( len(vertices) )
                          if NegativeROA=="0":
                              xy[regionID] = []
                              for vertex in vertices:
                                  # get the x value of the vertex / convert them into index in the tiled matrix of the base image
                                  # x = int(round(float(vertex.attributes['X'].value) / ImgMaxSizeX_orig * (cols*Img_Fact)))
                                  # y = int(round(float(vertex.attributes['Y'].value) / ImgMaxSizeY_orig * (rows*Img_Fact)))
                                  x = int(round(float(vertex.attributes['X'].value) / NewFact))
                                  y = int(round(float(vertex.attributes['Y'].value) / NewFact))
                                  xy[regionID].append((x,y))
                                  #print(vertex.attributes['X'].value, vertex.attributes['Y'].value, x, y )
    
                          elif NegativeROA=="1":
                              xy_neg[regionID] = []
                              for vertex in vertices:
                                  # get the x value of the vertex / convert them into index in the tiled matrix of the base image
                                  # x = int(round(float(vertex.attributes['X'].value) / ImgMaxSizeX_orig * (cols*Img_Fact)))
                                  # y = int(round(float(vertex.attributes['Y'].value) / ImgMaxSizeY_orig * (rows*Img_Fact)))
                                  x = int(round(float(vertex.attributes['X'].value) / NewFact))
                                  y = int(round(float(vertex.attributes['Y'].value) / NewFact))
                                  xy_neg[regionID].append((x,y))
    

                          #xy_a = np.array(xy[regionID])
        ## End Aperio
        elif AnnotationMode == 'QuPath':
          print("QuPath annotation file detected")
          xmlcontent = json.load(open(xmldir))
          xml_valid = True
          xy = {}
          xy_neg = {}
          NbRg = 0
          if 'annotations' in xmlcontent.keys():
            for eachR in range(len(xmlcontent['annotations'])):
              if 'class' in xmlcontent['annotations'][eachR].keys(): 
                labeltag = xmlcontent['annotations'][eachR]['class']
                # print(labeltag, "****", Attribute_Name)
                if (Attribute_Name==[]) | (Attribute_Name==''):
                  # No filter on label name
                  isLabelOK = True
                elif (Attribute_Name == labeltag):
                  isLabelOK = True
                elif Attribute_Name == "non_selected_regions":
                  isLabelOK = True
                else:
                  isLabelOK = False
                # print(isLabelOK)
                if isLabelOK:
                  regionID = str(NbRg)
                  xy[regionID] = []
                  vertices = xmlcontent['annotations'][eachR]['points']
                  NbRg += 1
                  xy[regionID] = [ii / NewFact for ii in vertices]
                  # no field for "negative region" - if it is, create a "xy_neg[regionID]"
 
  
	
#### Remove 2 spaces ### 
        # print("%d xy" % len(xy))
        #print(xy)
        # print("%d xy_neg"  % len(xy_neg))
        #print(xy_neg)
        # print("Img_Fact:")
        # print(NewFact)
        # print(ImgMaxSizeX_orig/NewFact, ImgMaxSizeY_orig/NewFact)
        # img = Image.new('L', (int(cols*Img_Fact), int(rows*Img_Fact)), 0)
        img = Image.new('L', (int(ImgMaxSizeX_orig/NewFact), int(ImgMaxSizeY_orig/NewFact)), 0)
        for regionID in xy.keys():
            xy_a = xy[regionID]
            # print(xy_a)
            ImageDraw.Draw(img,'L').polygon(xy_a, outline=255, fill=255)
        for regionID in xy_neg.keys():
            xy_a = xy_neg[regionID]
            ImageDraw.Draw(img,'L').polygon(xy_a, outline=255, fill=0)
        #img = img.resize((cols,rows), Image.ANTIALIAS)
        mask = np.array(img)
        #print(mask.shape)
        if Attribute_Name == "non_selected_regions":
           # scipy.misc.toimage(255-mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg"))
           Image.fromarray(255-mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg"))
        else:
           if self._mask_type==0:
             # scipy.misc.toimage(255-mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + "_inv.jpeg"))
             Image.fromarray(255-mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + "_inv.jpeg"))
           else:
             # scipy.misc.toimage(mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg"))
             Image.fromarray(mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg"))  
        #print(mask)
        return mask / 255.0, xml_valid, NewFact
        # Img_Fact


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, format, tile_size, overlap,
                limit_bounds, quality, workers, with_viewer, Bkg, basenameJPG, xmlfile, mask_type, ROIpc, oLabel, ImgExtension, SaveMasks, Mag, normalize, Fieldxml, pixelsize, pixelsizerange):
        if with_viewer:
            # Check extra dependency before doing a bunch of work
            import jinja2
        #print("line226 - %s " % (slidepath) )
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._xmlfile = xmlfile
        self._mask_type = mask_type
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._with_viewer = with_viewer
        self._Bkg = Bkg
        self._ROIpc = ROIpc
        self._dzi_data = {}
        self._xmlLabel = oLabel
        self._ImgExtension = ImgExtension
        self._SaveMasks = SaveMasks
        self._Mag = Mag
        self._normalize = normalize
        self._Fieldxml = Fieldxml
        self._pixelsize = pixelsize
        self._pixelsizerange = pixelsizerange
        self._rescale = False
        self._resize_ratio = 1
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                limit_bounds, quality, self._Bkg, self._ROIpc).start()

    def run(self):
        self._run_image()
        if self._with_viewer:
            for name in self._slide.associated_images:
                self._run_image(name)
            self._write_html()
            self._write_static()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            if self._with_viewer:
                 basename = os.path.join(self._basename, VIEWER_SLIDE_NAME)
            else:
                basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        # print("enter DeepZoomGenerator")
        if (self._Mag <= 0) and (self._pixelsizerange < 0):
               # calculate the best window size before rescaling to reach desired final pizelsize
               try:
                   Objective = float(self._slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
                   OrgPixelSizeX = float(self._slide.properties['openslide.mpp-x'])
                   OrgPixelSizeY = float(self._slide.properties['openslide.mpp-y'])
               except:
                   print("Error: No information found in the header")
                   return
               Desired_FoV_um = self._pixelsize * self._tile_size
               AllPxSizes = [OrgPixelSizeX * pow(2,nn) for nn in range(0,12)]
               AllBoxSizes = [round(Desired_FoV_um / (OrgPixelSizeX * pow(2,nn))) for nn in range(0,12)]
               for nn in range(0,12):
                   if AllBoxSizes[nn] < self._tile_size:
                       AllBoxSizes[nn] = 2000000
               Final_pixel_size_Diff = [abs(AllBoxSizes[x] / self._tile_size * AllPxSizes[x] - self._pixelsize) for x in range(0,12)]
               Best_level = [index for index, value in enumerate(Final_pixel_size_Diff) if value == min(Final_pixel_size_Diff)][-1]
               Adj_WindowSize = AllBoxSizes[Best_level]
               dz = DeepZoomGenerator(image, Adj_WindowSize, self._overlap,limit_bounds=self._limit_bounds)
               self._resize_ratio = float(Adj_WindowSize) / float(self._tile_size)
               print("info: Objective:" + str(Objective) + "; OrgPixelSizeX" + str(OrgPixelSizeX) + "; Desired_FoV_um: " + str(Desired_FoV_um) +"; Best_level: "+ str(Best_level) + "; resize_ratio: " +str(self._resize_ratio) + "; Adj_WindowSize:" + str(Adj_WindowSize) + "; self._tile_size: " + str(self._tile_size))
               #with open(os.path.join('/',join(self._basename.split('/')[:-1]), 'pixelsizes.txt')  , 'a') as file_out:
               #    file_out.write(self._basenameJPG + "\t" + str(AllPxSizes[Best_level]) + "\t" + str(AllPxSizes[Best_level]*resize_ratio))     
 
        else:
            dz = DeepZoomGenerator(image, self._tile_size, self._overlap,limit_bounds=self._limit_bounds)
            Best_level = -1
            self._resize_ratio = 1
            Adj_WindowSize = self._tile_size
        tiler = DeepZoomImageTiler(dz, basename, self._format, associated,self._queue, self._slide, self._basenameJPG, self._xmlfile, self._mask_type, self._xmlLabel, self._ROIpc, self._ImgExtension, self._SaveMasks, self._Mag, self._normalize, self._Fieldxml, self._pixelsize, self._pixelsizerange, Best_level, self._resize_ratio, Adj_WindowSize)
        try:
            tiler.run()
        except:
            print("Error in tiler.run(); image " + self._basenameJPG + " not processed")
        self._dzi_data[self._url_for(associated)] = tiler.get_dzi()



    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _write_html(self):
        import jinja2
        env = jinja2.Environment(loader=jinja2.PackageLoader(__name__),autoescape=True)
        template = env.get_template('slide-multipane.html')
        associated_urls = dict((n, self._url_for(n))
                for n in self._slide.associated_images)
        try:
            mpp_x = self._slide.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = self._slide.properties[openslide.PROPERTY_NAME_MPP_Y]
            mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            mpp = 0
        # Embed the dzi metadata in the HTML to work around Chrome's
        # refusal to allow XmlHttpRequest from file:///, even when
        # the originating page is also a file:///
        data = template.render(slide_url=self._url_for(None),slide_mpp=mpp,associated=associated_urls, properties=self._slide.properties, dzi_data=json.dumps(self._dzi_data))
        with open(os.path.join(self._basename, 'index.html'), 'w') as fh:
            fh.write(data)

    def _write_static(self):
        basesrc = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'static')
        basedst = os.path.join(self._basename, 'static')
        self._copydir(basesrc, basedst)
        self._copydir(os.path.join(basesrc, 'images'),
                os.path.join(basedst, 'images'))

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()



def ImgWorker(queue):
	# print("ImgWorker started")
	while True:
		cmd = queue.get()			
		if cmd is None:
			queue.task_done()
			break
		# print("Execute: %s" % (cmd))
		subprocess.Popen(cmd, shell=True).wait()
		queue.task_done()

def xml_read_labels(xmldir, Fieldxml, AnnotationMode):
        if AnnotationMode == 'Aperio':
          try:
              xmlcontent = minidom.parse(xmldir)
              xml_valid = True
          except:
              xml_valid = False
              print("error with minidom.parse(xmldir)")
              return [], xml_valid
          labeltag = xmlcontent.getElementsByTagName('Attribute')
          xml_labels = []
          for xmllabel in labeltag:
              xml_labels.append(xmllabel.attributes[Fieldxml].value)
              #xml_labels.append(xmllabel.attributes['Name'].value)
              # xml_labels.append(xmllabel.attributes['Value'].value)
          if xml_labels==[]:
              xml_labels = ['']
          # print(xml_labels)
        elif AnnotationMode == 'QuPath':
          data = json.load(open(xmldir))
          xml_labels = []
          xml_valid = False
          for eachR in range(len(data['annotations'])):
            xml_labels.append(data['annotations'][eachR]['class'])
          xml_labels = np.unique(xml_labels)
          if len(xml_labels) > 0:
            xml_valid = True
        return xml_labels, xml_valid 


if __name__ == '__main__':
	parser = OptionParser(usage='Usage: %prog [options] <slide>')

	parser.add_option('-L', '--ignore-bounds', dest='limit_bounds',
		default=True, action='store_false',
		help='display entire scan area')
	parser.add_option('-e', '--overlap', metavar='PIXELS', dest='overlap',
		type='int', default=1,
		help='overlap of adjacent tiles [1]')
	parser.add_option('-f', '--format', metavar='{jpeg|png}', dest='format',
		default='jpeg',
		help='image format for tiles [jpeg]')
	parser.add_option('-j', '--jobs', metavar='COUNT', dest='workers',
		type='int', default=4,
		help='number of worker processes to start [4]')
	parser.add_option('-o', '--output', metavar='NAME', dest='basename',
		help='base name of output file')
	parser.add_option('-Q', '--quality', metavar='QUALITY', dest='quality',
		type='int', default=90,
		help='JPEG compression quality [90]')
	parser.add_option('-r', '--viewer', dest='with_viewer',
		action='store_true',
		help='generate directory tree with HTML viewer')
	parser.add_option('-s', '--size', metavar='PIXELS', dest='tile_size',
		type='int', default=254,
		help='tile size [254]')
	parser.add_option('-B', '--Background', metavar='PIXELS', dest='Bkg',
		type='float', default=50,
		help='Max background threshold [50]; percentager of background allowed')
	parser.add_option('-x', '--xmlfile', metavar='NAME', dest='xmlfile',
		help='path to xml file from Aperio annotation; json for QuPath')
	parser.add_option('-F', '--Fieldxml', metavar='{Name|Value}', dest='Fieldxml',
		default='Value',
		help='which field of the xml file is the label saved')
	parser.add_option('-m', '--mask_type', metavar='COUNT', dest='mask_type',
		type='int', default=1,
		help='if xml file is used, keep tile within the ROI (1) or outside of it (0)')
	parser.add_option('-R', '--ROIpc', metavar='PIXELS', dest='ROIpc',
		type='float', default=50,
		help='To be used with xml file - minimum percentage of tile covered by ROI (white)')
	parser.add_option('-l', '--oLabelref', metavar='NAME', dest='oLabelref',
		help='To be used with xml file - Only tile for label which contains the characters in oLabel')
	parser.add_option('-S', '--SaveMasks', metavar='NAME', dest='SaveMasks',
		default=False,
		help='set to yes if you want to save ALL masks for ALL tiles (will be saved in same directory with <mask> suffix)')
	parser.add_option('-t', '--tmp_dcm', metavar='NAME', dest='tmp_dcm',
		help='base name of output folder to save intermediate dcm images converted to jpg (we assume the patient ID is the folder name in which the dcm images are originally saved)')
	parser.add_option('-M', '--Mag', metavar='PIXELS', dest='Mag',
		type='float', default=-1,
		help='Magnification at which tiling should be done; if Mag=-1 and pixelsize=-1, they will be tiles at all magnifications; if Mag=-1 and pixelsize>0, it will be tiled at a certain pixelsize')
	parser.add_option('-P', '--pixelsize',  metavar='PIXELS', dest='pixelsize',
		type='float', default=-1,
		help='Ignored if -1 or if Mag>0. Otherwise, will tile the svs files at the requested pixelsize (assumed to be stored in the header of the svs, in the openslide.mpp-x field')
	parser.add_option('-p', '--pixelsizerange',  metavar='PIXELS', dest='pixelsizerange',
		type='float', default=0,
		help='svs are always tiled at a factor of 2 from the original higher magnification. This is the range allowed around the pixelsize. svs will be tiles at whatever magnification is the closest to the pixelsize, plus or minus the deviation. Nothing will be tiled if no magnification falls within that range. If its valiue is -1, then the tiles will be rescaled to match the desired final size and pixelsize')
	parser.add_option('-N', '--normalize', metavar='NAME', dest='normalize',
		help='if normalization is needed, N list the mean and std for each channel. For example \'57,22,-8,20,10,5\' with the first 3 numbers being the targeted means, and then the targeted stds')




	(opts, args) = parser.parse_args()


	try:
		slidepath = args[0]
	except IndexError:
		parser.error('Missing slide argument')
	if opts.basename is None:
		opts.basename = os.path.splitext(os.path.basename(slidepath))[0]
	if opts.xmlfile is None:
		opts.xmlfile = ''

	try:
		if opts.normalize is not None:
			opts.normalize = [float(x) for x in opts.normalize.split(',')]
			if len(opts.normalize) != 6:
				opts.normalize = ''
				parser.error("ERROR: NO NORMALIZATION APPLIED: input vector does not have the right length - 6 values expected")
		else:
			opts.normalize  = ''

	except:
		opts.normalize = ''
		parser.error("ERROR: NO NORMALIZATION APPLIED: input vector does not have the right format")
            



	# get  images from the data/ file.
	files = glob(slidepath)  
	#ImgExtension = os.path.splitext(slidepath)[1]
	ImgExtension = slidepath.split('*')[-1]
	#files
	#len(files)
	# print(args)
	# print(args[0])
	# print(slidepath)
	# print(files)
	# print("***********************")

	'''
	dz_queue = JoinableQueue()
	procs = []
	print("Nb of processes:")
	print(opts.max_number_processes)
	for i in range(opts.max_number_processes):
		p = Process(target = ImgWorker, args = (dz_queue,))
		#p.deamon = True
		p.setDaemon = True
		p.start()
		procs.append(p)
	'''
	files = sorted(files)
	for imgNb in range(len(files)):
		filename = files[imgNb]
		#print(filename)
		opts.basenameJPG = os.path.splitext(os.path.basename(filename))[0]
		print("processing: " + opts.basenameJPG + " with extension: " + ImgExtension)
		#opts.basenameJPG = os.path.splitext(os.path.basename(slidepath))[0]
		#if os.path.isdir("%s_files" % (basename)):
		#	print("EXISTS")
		#else:
		#	print("Not Found")

		if ("dcm" in ImgExtension) :
			print("convert %s dcm to jpg" % filename)
			if opts.tmp_dcm is None:
				parser.error('Missing output folder for dcm>jpg intermediate files')
			elif not os.path.isdir(opts.tmp_dcm):
				parser.error('Missing output folder for dcm>jpg intermediate files')

			if filename[-3:] == 'jpg':
                            continue
			ImageFile=dicom.read_file(filename)
			im1 = ImageFile.pixel_array
			maxVal = float(im1.max())
			minVal = float(im1.min())
			height = im1.shape[0]
			width = im1.shape[1]
			image = np.zeros((height,width,3), 'uint8')
			image[...,0] = ((im1[:,:].astype(float) - minVal)  / (maxVal - minVal) * 255.0).astype(int)
			image[...,1] = ((im1[:,:].astype(float) - minVal)  / (maxVal - minVal) * 255.0).astype(int)
			image[...,2] = ((im1[:,:].astype(float) - minVal)  / (maxVal - minVal) * 255.0).astype(int)
			# dcm_ID = os.path.basename(os.path.dirname(filename))
			# opts.basenameJPG = dcm_ID + "_" + opts.basenameJPG
			filename = os.path.join(opts.tmp_dcm, opts.basenameJPG + ".jpg")
			# print(filename)
			imsave(filename,image)

			output = os.path.join(opts.basename, opts.basenameJPG)

			try:
				DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml, opts.pixelsize, opts.pixelsizerange).run()
			except Exception as e:
				print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
				print(e)

		#elif ("jpg" in ImgExtension) :
		#	output = os.path.join(opts.basename, opts.basenameJPG)
		#	if os.path.exists(output + "_files"):
		#		print("Image %s already tiled" % opts.basenameJPG)
		#		continue

		#	DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml, opts.pixelsize, opts.pixelsizerange).run()

		elif opts.xmlfile != '':
			# Check if Aperio or Qupath annotations
			# if len(glob(os.path.join(xmlfile,'*.xml'))) == 0:
			#	if len(glob(os.path.join(xmlfile,'*.json'))) == 0:
			#		print("Error: No xml or json file found for the annotations")
			xmldir = os.path.join(opts.xmlfile, opts.basenameJPG + '.xml')
			jsondir = os.path.join(opts.xmlfile, opts.basenameJPG + '.json')
				
			# print("xml:")
			# print(xmldir)
			if os.path.isfile(xmldir) | os.path.isfile(jsondir):
				if os.path.isfile(xmldir):
					AnnotationMode = 'Aperio'
				elif os.path.isfile(jsondir):
					AnnotationMode = 'QuPath'
					xmldir = jsondir
				if (opts.mask_type==1) or (opts.oLabelref!=''):
					# either mask inside ROI, or mask outside but a reference label exist
					xml_labels, xml_valid = xml_read_labels(xmldir, opts.Fieldxml, AnnotationMode)
					if (opts.mask_type==1):
						# No inverse mask
						Nbr_ROIs_ForNegLabel = 1
					elif (opts.oLabelref!=''):
						# Inverse mask and a label reference exist
						Nbr_ROIs_ForNegLabel = 0

					for oLabel in xml_labels:
						# print("label is %s and ref is %s" % (oLabel, opts.oLabelref))
						if (opts.oLabelref in oLabel) or (opts.oLabelref==''):
							# is a label is identified 
							if (opts.mask_type==0):
								# Inverse mask and label exist in the image
								Nbr_ROIs_ForNegLabel += 1
								# there is a label, and map is to be inverted
								output = os.path.join(opts.basename, oLabel+'_inv', opts.basenameJPG)
								if not os.path.exists(os.path.join(opts.basename, oLabel+'_inv')):
									os.makedirs(os.path.join(opts.basename, oLabel+'_inv'))
							else:
								Nbr_ROIs_ForNegLabel += 1
								output = os.path.join(opts.basename, oLabel, opts.basenameJPG)
								if not os.path.exists(os.path.join(opts.basename, oLabel)):
									os.makedirs(os.path.join(opts.basename, oLabel))
							if 1:
							#try:
								DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml, opts.pixelsize, opts.pixelsizerange).run()
							#except:
							#	print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
						if Nbr_ROIs_ForNegLabel==0:
							print("label %s is not in that image; invert everything" % (opts.oLabelref))
							# a label ref was given, and inverse mask is required but no ROI with this label in that map --> take everything
							oLabel = opts.oLabelref
							output = os.path.join(opts.basename, opts.oLabelref+'_inv', opts.basenameJPG)
							if not os.path.exists(os.path.join(opts.basename, oLabel+'_inv')):
								os.makedirs(os.path.join(opts.basename, oLabel+'_inv'))
							if 1:
							#try:
								DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml, opts.pixelsize, opts.pixelsizerange).run()
							#except:
							#	print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))

				else:
					# Background
					oLabel = "non_selected_regions"
					output = os.path.join(opts.basename, oLabel, opts.basenameJPG)
					if not os.path.exists(os.path.join(opts.basename, oLabel)):
						os.makedirs(os.path.join(opts.basename, oLabel))
					try:
						DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml, opts.pixelsize, opts.pixelsizerange).run()
					except Exception as e:
						print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
						print(e)

			else:
				if (ImgExtension == ".jpg") | (ImgExtension == ".dcm") :
					print("Input image to be tiled is jpg or dcm and not svs - will be treated as such")
					output = os.path.join(opts.basename, opts.basenameJPG)
					try:
						DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml, opts.pixelsize, opts.pixelsizerange).run()
					except Exception as e:
						print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
						print(e)


				else:
					print("No xml file found for slide %s.svs (expected: %s). Directory or xml file does not exist" %  (opts.basenameJPG, xmldir) )
					continue
		else:
			output = os.path.join(opts.basename, opts.basenameJPG)
			if os.path.exists(output + "_files"):
				print("Image %s already tiled" % opts.basenameJPG)
				continue
			# try:
			if True:
				DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize, opts.Fieldxml, opts.pixelsize, opts.pixelsizerange).run()
			#except Exception as e:
			#	print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
			#	print(e)
	'''
	dz_queue.join()
	for i in range(opts.max_number_processes):
		dz_queue.put( None )
	'''

	print("End")









