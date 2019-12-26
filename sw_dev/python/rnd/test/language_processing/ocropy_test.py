#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/tmbdev/ocropy

import os, cv2
import ocrolib
from ocrolib import psegutils, morph, sl
from ocrolib.exceptions import OcropusException
from ocrolib.toplevel import *
from scipy.ndimage import filters, interpolation, morphology, measurements
from scipy.ndimage.filters import gaussian_filter, uniform_filter, maximum_filter
from scipy import stats

def print_info(objs):
	print("INFO: ", objs)

def estimate_skew_angle(image,angles,debug):
	estimates = []
	for a in angles:
		v = np.mean(interpolation.rotate(image,a,order=0,mode='constant'),axis=1)
		v = np.var(v)
		estimates.append((v,a))
	if debug>0:
		plt.plot([y for x,y in estimates],[x for x,y in estimates])
		plt.ginput(1,debug)
	_,a = max(estimates)
	return a

def normalize_raw_image(raw):
	''' perform image normalization '''
	image = raw-np.amin(raw)
	if np.amax(image)==np.amin(image):
		print_info("# image is empty: %s" % (fname))
		return None
	image /= np.amax(image)
	return image

def estimate_local_whitelevel(image, zoom=0.5, perc=80, range=20, debug=0):
	'''flatten it by estimating the local whitelevel
	zoom for page background estimation, smaller=faster, default: %(default)s
	percentage for filters, default: %(default)s
	range for filters, default: %(default)s
	'''
	m = interpolation.zoom(image,zoom)
	m = filters.percentile_filter(m,perc,size=(range,2))
	m = filters.percentile_filter(m,perc,size=(2,range))
	m = interpolation.zoom(m,1.0/zoom)
	if debug>0:
		plt.clf()
		plt.imshow(m,vmin=0,vmax=1)
		plt.ginput(1,debug)
	w,h = np.minimum(np.array(image.shape),np.array(m.shape))
	flat = np.clip(image[:w,:h]-m[:w,:h]+1,0,1)
	if debug>0:
		plt.clf()
		plt.imshow(flat,vmin=0,vmax=1)
		plt.ginput(1,debug)
	return flat

def estimate_skew(flat, bignore=0.1, maxskew=2, skewsteps=8, debug=0):
	''' estimate skew angle and rotate'''
	d0,d1 = flat.shape
	o0,o1 = int(bignore*d0),int(bignore*d1) # border ignore
	flat = np.amax(flat)-flat
	flat -= np.amin(flat)
	est = flat[o0:d0-o0,o1:d1-o1]
	ma = maxskew
	ms = int(2*maxskew*skewsteps)
	# print(linspace(-ma,ma,ms+1))
	angle = estimate_skew_angle(est,np.linspace(-ma,ma,ms+1), debug)
	flat = interpolation.rotate(flat,angle,mode='constant',reshape=0)
	flat = np.amax(flat)-flat
	return flat, angle

def estimate_thresholds(flat, bignore=0.1, escale=1.0, lo=5, hi=90, debug=0):
	'''# estimate low and high thresholds
	ignore this much of the border for threshold estimation, default: %(default)s
	scale for estimating a mask over the text region, default: %(default)s
	lo percentile for black estimation, default: %(default)s
	hi percentile for white estimation, default: %(default)s
	'''
	d0,d1 = flat.shape
	o0,o1 = int(bignore*d0),int(bignore*d1)
	est = flat[o0:d0-o0,o1:d1-o1]
	if escale>0:
		# by default, we use only regions that contain
		# significant variance; this makes the percentile
		# based low and high estimates more reliable
		e = escale
		v = est-filters.gaussian_filter(est,e*20.0)
		v = filters.gaussian_filter(v**2,e*20.0)**0.5
		v = (v>0.3*np.amax(v))
		v = morphology.binary_dilation(v,structure=np.ones((int(e*50),1)))
		v = morphology.binary_dilation(v,structure=np.ones((1,int(e*50))))
		if debug>0:
			plt.imshow(v)
			plt.ginput(1,debug)
		est = est[v]
	lo = stats.scoreatpercentile(est.ravel(),lo)
	hi = stats.scoreatpercentile(est.ravel(),hi)
	return lo, hi

def find(condition):
	"Return the indices where ravel(condition) is true"
	res, = np.nonzero(np.ravel(condition))
	return res

def DSAVE(title,image,debug=0):
	if not debug: return
	if type(image)==list:
		assert len(image)==3
		image = np.transpose(np.array(image),[1,2,0])
	fname = "_"+title+".png"
	print_info("debug " + fname)
	imsave(fname,image.astype('float'))

def compute_separators_morph(binary,scale,sepwiden,maxseps):
	"""Finds vertical black lines corresponding to column separators."""
	d0 = int(max(5,scale/4))
	d1 = int(max(5,scale))+sepwiden
	thick = morph.r_dilation(binary,(d0,d1))
	vert = morph.rb_opening(thick,(10*scale,1))
	vert = morph.r_erosion(vert,(d0//2,sepwiden))
	vert = morph.select_regions(vert,sl.dim1,min=3,nbest=2*maxseps)
	vert = morph.select_regions(vert,sl.dim0,min=20*scale,nbest=maxseps)
	return vert

def compute_colseps_conv(binary,csminheight,maxcolseps,scale=1.0):
	"""Find column separators by convolution and
	thresholding."""
	h,w = binary.shape
	# find vertical whitespace by thresholding
	smoothed = gaussian_filter(1.0*binary,(scale,scale*0.5))
	smoothed = uniform_filter(smoothed,(5.0*scale,1))
	thresh = (smoothed<np.amax(smoothed)*0.1)
	DSAVE("1thresh",thresh)
	# find column edges by filtering
	grad = gaussian_filter(1.0*binary,(scale,scale*0.5),order=(0,1))
	grad = uniform_filter(grad,(10.0*scale,1))
	# grad = abs(grad) # use this for finding both edges
	grad = (grad>0.5*np.amax(grad))
	DSAVE("2grad",grad)
	# combine edges and whitespace
	seps = np.minimum(thresh,maximum_filter(grad,(int(scale),int(5*scale))))
	seps = maximum_filter(seps,(int(2*scale),1))
	DSAVE("3seps",seps)
	# select only the biggest column separators
	seps = morph.select_regions(seps,sl.dim0,min=csminheight*scale,nbest=maxcolseps)
	DSAVE("4seps",seps)
	return seps

def compute_colseps(binary,scale,blackseps,maxseps,maxcolseps,csminheight,sepwiden):
	"""Computes column separators either from vertical black lines or whitespace."""
	print_info("considering at most %g whitespace column separators" % maxcolseps)
	colseps = compute_colseps_conv(binary,csminheight,maxcolseps,scale)
	DSAVE("colwsseps",0.7*colseps+0.3*binary)
	if blackseps and maxseps == 0:
		# simulate old behaviour of blackseps when the default value
		# for maxseps was 2, but only when the maxseps-value is still zero
		# and not set manually to a non-zero value
		maxseps = 2
	if maxseps > 0:
		print_info("considering at most %g black column separators" % maxseps)
		seps = compute_separators_morph(binary,scale,sepwiden,maxseps)
		DSAVE("colseps",0.7*seps+0.3*binary)
		#colseps = compute_colseps_morph(binary,scale)
		colseps = np.maximum(colseps,seps)
		binary = np.minimum(binary,1-seps)
	binary,colseps = apply_mask(binary,colseps)
	return colseps,binary

def apply_mask(binary,colseps):
	try:
		#mask = ocrolib.read_image_binary(base+".mask.png")
		mask = ocrolib.read_image_binary("./ocropy_test.mask.png")
	except IOError:
		return binary,colseps
	masked_seps = np.maximum(colseps,mask)
	binary = np.minimum(binary,1-masked_seps)
	DSAVE("masked_seps", masked_seps)
	return binary,masked_seps

def compute_gradmaps(binary,scale,usegauss,vscale,hscale):
    # use gradient filtering to find baselines
    boxmap = psegutils.compute_boxmap(binary,scale)
    cleaned = boxmap*binary
    DSAVE("cleaned",cleaned)
    if usegauss:
        # this uses Gaussians
        grad = gaussian_filter(1.0*cleaned,(vscale*0.3*scale,
                                            hscale*6*scale),order=(1,0))
    else:
        # this uses non-Gaussian oriented filters
        grad = gaussian_filter(1.0*cleaned,(max(4,vscale*0.3*scale),
                                            hscale*scale),order=(1,0))
        grad = uniform_filter(grad,(vscale,hscale*6*scale))
    bottom = ocrolib.norm_max((grad<0)*(-grad))
    top = ocrolib.norm_max((grad>0)*grad)
    return bottom,top,boxmap

def compute_segmentation(binary,scale,blackseps,maxseps,maxcolseps,csminheight,sepwiden,usegauss,vscale,hscale,threshold,quiet):
	"""Given a binary image, compute a complete segmentation into
	lines, computing both columns and text lines."""
	binary = np.array(binary,'B')

	# start by removing horizontal black lines, which only
	# interfere with the rest of the page segmentation
	binary = remove_hlines(binary,scale)

	# do the column finding
	if not quiet: print_info("computing column separators")
	colseps,binary = compute_colseps(binary,scale,blackseps,maxseps,maxcolseps,csminheight,sepwiden)

	# now compute the text line seeds
	if not quiet: print_info("computing lines")
	bottom,top,boxmap = compute_gradmaps(binary,scale,usegauss,vscale,hscale)
	seeds = compute_line_seeds(binary,bottom,top,colseps,scale,threshold,vscale)
	DSAVE("seeds",[bottom,top,boxmap])

	# spread the text line seeds to all the remaining
	# components
	if not quiet: print_info("propagating labels")
	llabels = morph.propagate_labels(boxmap,seeds,conflict=0)
	if not quiet: print_info("spreading labels")
	spread = morph.spread_labels(seeds,maxdist=scale)
	llabels = np.where(llabels>0,llabels,spread*binary)
	segmentation = llabels*binary
	return segmentation

def compute_line_seeds(binary,bottom,top,colseps,scale,threshold,vscale):
	"""Base on gradient maps, computes candidates for baselines
	and xheights.  Then, it marks the regions between the two
	as a line seed."""
	t = threshold
	vrange = int(vscale*scale)
	bmarked = maximum_filter(bottom==maximum_filter(bottom,(vrange,0)),(2,2))
	bmarked = bmarked*(bottom>t*np.amax(bottom)*t)*(1-colseps)
	tmarked = maximum_filter(top==maximum_filter(top,(vrange,0)),(2,2))
	tmarked = tmarked*(top>t*np.amax(top)*t/2)*(1-colseps)
	tmarked = maximum_filter(tmarked,(1,20))
	seeds = np.zeros(binary.shape,'i')
	delta = max(3,int(scale/2))
	for x in range(bmarked.shape[1]):
		transitions = sorted([(y,1) for y in find(bmarked[:,x])]+[(y,0) for y in find(tmarked[:,x])])[::-1]
		transitions += [(0,0)]
		for l in range(len(transitions)-1):
			y0,s0 = transitions[l]
			if s0==0: continue
			seeds[y0-delta:y0,x] = 1
			y1,s1 = transitions[l+1]
			if s1==0 and (y0-y1)<5*scale: seeds[y1:y0,x] = 1
	seeds = maximum_filter(seeds,(1,int(1+scale)))
	seeds = seeds*(1-colseps)
	DSAVE("lineseeds",[seeds,0.3*tmarked+0.7*bmarked,binary])
	seeds,_ = morph.label(seeds)
	return seeds

def remove_hlines(binary,scale,maxsize=10):
	labels,_ = morph.label(binary)
	objects = morph.find_objects(labels)
	for i,b in enumerate(objects):
		if sl.width(b)>maxsize*scale:
			labels[b][labels[b]==i+1] = 0
	return np.array(labels!=0,'B')

# REF [file] >> ${OCROPY_HOME}/ocropus-nlbin
def binarize(image_filepath):
	raw = ocrolib.read_image_gray(image_filepath)

	# Perform image normalization.
	image = normalize_raw_image(raw)

	threshold = 0.5  # Threshold, determines lightness.
	zoom = 0.5  # Zoom for page background estimation, smaller=faster.
	escale = 1.0  # Scale for estimating a mask over the text region.
	bignore = 0.1  # Ignore this much of the border for threshold estimation.
	perc = 80  # Percentage for filters.
	range = 20  # Range for filters.
	maxskew = 2  # Skew angle estimation parameters (degrees).
	lo = 5  # Percentile for black estimation.
	hi = 90  # Percentile for white estimation.
	skewsteps = 8  # Steps for skew angle estimation (per degree).
	debug = 0  # Display intermediate results.

	# Flatten it by estimating the local whitelevel.
	flat = estimate_local_whitelevel(image, zoom, perc, range, debug)

	# Estimate skew angle and rotate.
	flat, angle = estimate_skew(flat, bignore, maxskew, skewsteps, debug)

	# Estimate low and high thresholds.
	lo, hi = estimate_thresholds(flat, bignore, escale, lo, hi, debug)

	# Rescale the image to get the gray scale image.
	flat -= lo
	flat /= (hi - lo)
	flat = np.clip(flat, 0, 1)

	bin = 1 * (flat > threshold)

	if False:
		# Output the normalized grayscale and the thresholded images.
		ocrolib.write_image_binary('./ocropy_test.bin.png', bin)
		ocrolib.write_image_gray('./ocropy_test.nrm.png', flat)

	return bin, flat

# REF [file] >> ${OCROPY_HOME}/ocropus-gpageseg
def analyze_page_layout(binary, gray, rgb=None):
	hscale = 1.0  # Non-standard scaling of horizontal parameters.
	vscale = 1.0  # Non-standard scaling of vertical parameters.
	threshold = 0.2  # baseline threshold.
	usegauss = True  # Use gaussian instead of uniform.
	maxseps = 0  # Maximum black column separators.
	sepwiden = 10  # Widen black separators (to account for warping).
	blackseps = True
	maxcolseps = 3  # Maximum # whitespace column separators.
	csminheight = 10  # Minimum column height (units=scale).
	noise = 8  # Noise threshold for removing small components from lines.
	gray_output = True  # Output grayscale lines as well, which are extracted from the grayscale version of the pages.
	pad = 3  # Padding for extracted lines.
	expand = 3  # Expand mask for grayscale extraction.

	if False:
		bin_image_filepath = './ocropy_test.bin.png'
		gray_image_filepath = './ocropy_test.nrm.png'

		binary = ocrolib.read_image_binary(bin_image_filepath)
		gray = ocrolib.read_image_gray(gray_image_filepath)

	binary = 1 - binary  # Invert.

	scale = psegutils.estimate_scale(binary)
	segmentation = compute_segmentation(binary, scale, blackseps, maxseps, maxcolseps, csminheight, sepwiden, usegauss, vscale, hscale, threshold, quiet=True)

	lines = psegutils.compute_lines(segmentation, scale)
	order = psegutils.reading_order([l.bounds for l in lines])
	lsort = psegutils.topsort(order)

	# Renumber the labels so that they conform to the specs.
	nlabels = np.amax(segmentation) + 1
	renumber = np.zeros(nlabels, 'i')
	for i, v in enumerate(lsort): renumber[lines[v].label] = 0x010000 + (i + 1)
	segmentation = renumber[segmentation]  # Image.

	lines = [lines[i] for i in lsort]

	# Visualize bounding boxes.
	if False:
		if rgb is not None:
			# REF [function] >> extract_masked() in ${OCROPY_HOME}/ocrolib/psegutils.py.
			for l in lines:
				y0, x0, y1, x1 = [int(x) for x in [l.bounds[0].start, l.bounds[1].start, l.bounds[0].stop, l.bounds[1].stop]]
				cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)
			cv2.imshow('Image', rgb)
			cv2.waitKey(0)

	# Output everything.
	if False:
		if not os.path.exists(outputdir):
			os.mkdir(outputdir)

		ocrolib.write_page_segmentation("%s.pseg.png" % outputdir, segmentation)
		cleaned = ocrolib.remove_noise(binary, noise)
		for i, l in enumerate(lines):
			binline = psegutils.extract_masked(1 - cleaned, l, pad=pad, expand=expand)  # Image.
			ocrolib.write_image_binary("%s/01%04x.bin.png" % (outputdir, i + 1), binline)
			if gray_output:
				grayline = psegutils.extract_masked(gray, l, pad=pad, expand=expand)  # Image.
				ocrolib.write_image_gray("%s/01%04x.nrm.png" % (outputdir, i + 1), grayline)

# REF [file] >> ${OCROPY_HOME}/ocropus-rpred
def recognize():
	raise NotImplementedError

def simple_example():
	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/work/dataset'
	else:
		data_dir_path = 'D:/work/dataset'
	image_filepath = data_dir_path + '/text/receipt_epapyrus/keit_20190619/크기변환_카드영수증_5-1.png'
	#image_filepath = data_dir_path + '/text/receipt_epapyrus/epapyrus_20190618/receipt_1/img01.jpg'

	rgb = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
	if rgb is None:
		print('Failed to load an image file, {}.'.format(image_filepath))
		return

	binary, gray = binarize(image_filepath)	
	analyze_page_layout(binary, gray, rgb)
	#recognize()  # Not yet implemented.

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
