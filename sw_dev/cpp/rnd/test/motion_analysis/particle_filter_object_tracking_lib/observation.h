/** @file
    Definitions for observation model functions for player tracking
    
    @author Rob Hess
    @version 1.0.0-20060306
*/

#ifndef OBSERVATION_H
#define OBSERVATION_H

/******************************* Definitions *********************************/

/* number of bins of HSV in histogram */
#define NH 10
#define NS 10
#define NV 10

/* max HSV values */
#define H_MAX 360.0
#define S_MAX 1.0
#define V_MAX 1.0

/* low thresholds on saturation and value for histogramming */
#define S_THRESH 0.1
#define V_THRESH 0.2

/* distribution parameter */
#define LAMBDA 20

/******************************** Structures *********************************/

/**
   An HSV histogram represented by NH * NS + NV bins.  Pixels with saturation
   and value greater than S_THRESH and V_THRESH fill the first NH * NS bins.
   Other, "colorless" pixels fill the last NV value-only bins.
*/
typedef struct histogram {
  float histo[NH*NS + NV];   /**< histogram array */
  int n;                     /**< length of histogram array */
} histogram;


/*************************** Function Definitions ****************************/

/**
   Converts a BGR image to HSV colorspace

   @param img image to be converted

   @return Returns img converted to a 3-channel, 32-bit HSV image with
     S and V values in the range [0,1] and H value in the range [0,360]
*/
IplImage* bgr2hsv( IplImage* img );


/**
   Calculates the histogram bin into which an HSV entry falls
   
   @param h Hue
   @param s Saturation
   @param v Value
   
   @return Returns the bin index corresponding to the HSV color defined by
     \a h, \a s, and \a v.
*/
int histo_bin( float h, float s, float v );


/**
   Calculates a cumulative histogram as defined above for a given array
   of images
   
   @param imgs an array of images over which to compute a cumulative histogram;
     each must have been converted to HSV colorspace using bgr2hsv()
   @param n the number of images in imgs
   
   @return Returns an un-normalized HSV histogram for \a imgs
*/
histogram* calc_histogram( IplImage** imgs, int n );


/**
   Normalizes a histogram so all bins sum to 1.0
   
   @param histo a histogram
*/
void normalize_histogram( histogram* histo );


/**
   Computes squared distance metric based on the Battacharyya similarity
   coefficient between histograms.

   @param h1 first histogram; should be normalized
   @param h2 second histogram; should be normalized
   
   @return Rerns a squared distance based on the Battacharyya similarity
     coefficient between \a h1 and \a h2
*/
float histo_dist_sq( histogram* h1, histogram* h2 );


/**
   Computes the likelihood of there being a player at a given location in
   an image

   @param img image that has been converted to HSV colorspace using bgr2hsv()
   @param r row location of center of window around which to compute likelihood
   @param c col location of center of window around which to compute likelihood
   @param w width of region over which to compute likelihood
   @param h height of region over which to compute likelihood
   @param ref_histo reference histogram for a player; must have been
     normalized with normalize_histogram()

   @return Returns the likelihood of there being a player at location
     (\a r, \a c) in \a img
*/
float likelihood( IplImage* img, int r, int c,
		  int w, int h, histogram* ref_histo );


/**
   Returns an image containing the likelihood of there being a player at
   each pixel location in an image
   
   @param img the image for which likelihood is to be computed; should have
     been converted to HSV colorspace using bgr2hsv()
   @param w width of region over which to compute likelihood
   @param h height of region over which to compute likelihood
   @param ref_histo reference histogram for a player; must have been
     normalized with normalize_histogram()
     
   @return Returns a single-channel, 32-bit floating point image containing
     the likelihood of every pixel location in \a img normalized so that the
     sum of likelihoods is 1.
*/
IplImage* likelihood_image( IplImage* img, int w, int h, histogram* ref_histo);


/**
   Exports histogram data to a specified file.  The file is formatted as
   follows (intended for use with gnuplot:

   0 <h_0>
   ...
   <i> <h_i>
   ...
   <n> <h_n>

   Where n is the number of histogram bins and h_i, i = 1..n are
   floating point bin values

   @param histo histogram to be exported
   @param filename name of file to which histogram is to be exported

   @return Returns 1 on success or 0 on failure
*/
int export_histogram( histogram* histo, char* filename );



#endif
