'''
    File name: 0b_tileLoop_deepzoom.py
    Date created: March/2017
    Date last modified: 2/25/2017
    Python Version: 2.7 (native on the cluster


	Source:
	Tiling code comes from:
	from https://github.com/openslide/openslide-python/blob/master/examples/deepzoom/deepzoom_tile.py
	which is Copyright (c) 2010-2015 Carnegie Mellon University

	Objective:
	Tile svs images

	Be careful:
	Overload of the node - may have memory issue if node is shared with other jobs.

	Initial tests:
	tested on Test_20_tiled/Test2 and Test5  using imgExample = "/ifs/home/coudrn01/NN/Lung/Test_20imgs/*/*svs"

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
import subprocess
from glob import glob
from multiprocessing import Process, JoinableQueue
import time
import os
import sys


VIEWER_SLIDE_NAME = 'slide'


class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,quality, _Bkg):
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

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            #associated, level, address, outfile = data
            associated, level, address, outfile, format, outfile_bw = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            #try:
            if True:
                try:
                    tile = dz.get_tile(level, address)
                    # A single tile is being read
                    #nc added: check the percentage of the image with "information". Should be above 50%
                    gray = tile.convert('L')
                    bw = gray.point(lambda x: 0 if x<220 else 1, 'F')
                    arr = np.array(np.asarray(bw))
                    avgBkg = np.average(bw)
                    bw = gray.point(lambda x: 0 if x<220 else 1, '1')
                    #outfile = os.path.join(outfile, '%s.%s' % (str(round(avgBkg, 3)),format) )
                    #outfile_bw = os.path.join(outfile_bw, '%s.%s' % (str(round(avgBkg, 3)),format) )
                    # bw.save(outfile_bw, quality=self._quality)
                    if avgBkg < (self._Bkg / 100):
                        tile.save(outfile, quality=self._quality)
                        #print("%s good: %f" %(outfile, avgBkg))
                    #else:
                            #print("%s empty: %f" %(outfile, avgBkg))
                    self._queue.task_done()
                except:
                    print(level, address)
                    print("image %s failed at dz.get_tile for level %f" % (self._slidepath, level))
                    self._queue.task_done()

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, format, associated, queue, slide, basenameJPG):
        self._dz = dz
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._slide = slide

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
            print(self._basename + " - Obj information found")
        except:
            print(self._basename + " - No Obj information found")
            return
        #calculate magnifications
        Available = tuple(Objective / x for x in Factors)
        #find highest magnification greater than or equal to 'Desired'
        Mismatch = tuple(x-Magnification for x in Available)
        AbsMismatch = tuple(abs(x) for x in Mismatch)
        if len(AbsMismatch) < 1:
          print(self._basename + " - Objective field empty!")
          return
        if(min(AbsMismatch) <= tol):
            Level = int(AbsMismatch.index(min(AbsMismatch)))
            Factor = 1
        else: #pick next highest level, downsample
            Level = int(max([i for (i, val) in enumerate(Mismatch) if val > 0]))
            Factor = Magnification / Available[Level]
        # end added
        #for level in range(self._dz.level_count):
        for level in range(self._dz.level_count-1,-1,-1):
            ThisMag = Available[0]/pow(2,self._dz.level_count-(level+1))
            ########################################
            #tiledir = os.path.join("%s_files" % self._basename, str(level))
            tiledir = os.path.join("%s_files" % self._basename, str(ThisMag))
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)
            cols, rows = self._dz.level_tiles[level]
            for row in range(rows):
                for col in range(cols):
                    InsertBaseName = False
                    if InsertBaseName:
                      tilename = os.path.join(tiledir, '%s_%d_%d.%s' % (
                                      self._basenameJPG, col, row, self._format))
                      tilename_bw = os.path.join(tiledir, '%s_%d_%d_bw.%s' % (
                                      self._basenameJPG, col, row, self._format))
                    else:
                      tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                      col, row, self._format))
                      tilename_bw = os.path.join(tiledir, '%d_%d_bw.%s' % (
                                      col, row, self._format))


                    if not os.path.exists(tilename):
                        self._queue.put((self._associated, level, (col, row),
                                    tilename, self._format, tilename_bw))
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


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, format, tile_size, overlap,
                limit_bounds, quality, workers, with_viewer, Bkg, basenameJPG):
        if with_viewer:
            # Check extra dependency before doing a bunch of work
            import jinja2
        print("line226 - %s " % (slidepath) )
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._with_viewer = with_viewer
        self._Bkg = Bkg
        self._dzi_data = {}
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                limit_bounds, quality, self._Bkg).start()

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
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,limit_bounds=self._limit_bounds)
        tiler = DeepZoomImageTiler(dz, basename, self._format, associated,self._queue, self._slide, self._basenameJPG)
        tiler.run()
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
		help='Max background threshold [50]')





    	(opts, args) = parser.parse_args()


	try:
		slidepath = args[0]
	except IndexError:
		parser.error('Missing slide argument')
	if opts.basename is None:
		opts.basename = os.path.splitext(os.path.basename(slidepath))[0]

	

	# Initialization
	# imgExample = "/ifs/home/coudrn01/NN/Lung/RawImages/*/*svs"
	# tile_size = 512
	# max_number_processes = 10
	# NbrCPU = 4


	# get  images from the data/ file.
	files = glob(slidepath)  
	files
	len(files)
	# print(files)
	print("***********************")

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
	for imgNb in range(len(files)):
		filename = files[imgNb]
		#print(filename)
		opts.basenameJPG = os.path.splitext(os.path.basename(filename))[0]
		print("processing: " + opts.basenameJPG)
		#opts.basenameJPG = os.path.splitext(os.path.basename(slidepath))[0]
		#if os.path.isdir("%s_files" % (basename)):
		#	print("EXISTS")
		#else:
		#	print("Not Found")

		output = os.path.join(opts.basename, opts.basenameJPG)

		
		# dz_queue.put(DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG).run())
		DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG).run()
	'''
	dz_queue.join()
	for i in range(opts.max_number_processes):
		dz_queue.put( None )
	'''

	print("End")









