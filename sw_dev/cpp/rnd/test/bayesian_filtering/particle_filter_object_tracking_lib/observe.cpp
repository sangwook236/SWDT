/*
  Test obervation model functions for player tracking

  @author Rob Hess
  @version 1.0.0-20060306
*/

#include "defs.h"
#include "utils.h"
#include "observation.h"

/* command line options */
#define OPTIONS ":h"


/***************************** Function Prototypes ***************************/

void usage( char* );
void arg_parse( int, char** );


/****************************** Defs and Globals *****************************/

char* pname;                  /* program name */
IplImage* in_img;             /* input image */
IplImage** ref_imgs;          /* array of player reference images */
int num_ref_imgs;             /* count of player reference images */

/*********************************** Main ************************************/

int main( int argc, char** argv )
{
  IplImage* hsv_img;
  IplImage** hsv_ref_imgs;
  IplImage* l32f, * l;
  histogram* ref_histo;
  double max;
  int i;

  arg_parse( argc, argv );

  /* compute HSV histogram over all reference image */
  hsv_img = bgr2hsv( in_img );
  hsv_ref_imgs = (IplImage**)malloc( num_ref_imgs * sizeof( IplImage* ) );
  for( i = 0; i < num_ref_imgs; i++ )
    hsv_ref_imgs[i] = bgr2hsv( ref_imgs[i] );
  ref_histo = calc_histogram( hsv_ref_imgs, num_ref_imgs );
  normalize_histogram( ref_histo );

  /* compute likelihood at every pixel in input image */
  fprintf( stderr, "Computing likelihood... " );
  fflush( stderr );
  l32f = likelihood_image( hsv_img, ref_imgs[0]->width,
			   ref_imgs[0]->height, ref_histo );
  fprintf( stderr, "done\n");

  /* convert likelihood image to uchar and display */
  cvMinMaxLoc( l32f, NULL, &max, NULL, NULL, NULL );
  l = cvCreateImage( cvGetSize( l32f ), IPL_DEPTH_8U, 1 );
  cvConvertScale( l32f, l, 255.0 / max, 0 );
  cvNamedWindow( "likelihood", 1 );
  cvShowImage( "likelihood", l );
  cvNamedWindow( "image", 1 );
  cvShowImage( "image", in_img );
  cvWaitKey(0);
}


/************************** Function Definitions *****************************/

/* print usage for this program */
void usage( char* name )
{
  fprintf(stderr, "%s: compute liklihood of there being a football player" \
	  " at all points\n  in an image\n\n", name);
  fprintf(stderr, "Usage: %s [options] <img_file> <player_img> [...]\n\n",
	  name);
  fprintf(stderr, "Arguments:\n");
  fprintf(stderr, "  <img_file>          An image on which to compute " \
	  "likelihoods\n");
  fprintf(stderr, "  <ref_img> [...]     A list of files from which to build" \
	  " a reference histogram\n");
  fprintf(stderr, "\nOptions:\n");
  fprintf(stderr, "  -h                  Display this message and exit\n");
}



/*
  arg_parse() parses the command line arguments, setting appropriate globals.

  argc and argv should be passed directly from the command line
*/
void arg_parse( int argc, char** argv )
{
  int i = 0;
  /*extract program name from command line (remove path, if present) */
  pname = remove_path( argv[0] );

  /*parse commandline options */
  while( TRUE )
    {
      char* arg_check;
      int arg = getopt( argc, argv, OPTIONS );
      if( arg == -1 )
	break;

      switch( arg )
	{
	  /* user asked for help */
	case 'h':
	  usage( pname );
	  exit(0);
	  break;

	  /* catch invalid arguments */
	default:
	  fatal_error( "-%c: invalid option\nTry '%s -h' for help.",
		       optopt, pname );
	}
    }

  /* make sure input and output files are specified */
  if( argc - optind < 1 )
    fatal_error( "no input image specified.\nTry '%s -h' for help.", pname );
  if( argc - optind < 2 )
    fatal_error( "no reference images specified.\nTry '%s -h' for help.",
		 pname );

  /* import input image */
  in_img = cvLoadImage( argv[optind], 1 );
  if( ! in_img )
    fatal_error("error importing image from %s", argv[optind]);

  /* import reference images */
  num_ref_imgs = argc - ++optind;
  ref_imgs = (IplImage **)malloc( num_ref_imgs * sizeof( IplImage*) );
  while( optind < argc )
    {
      ref_imgs[i] = cvLoadImage( argv[optind], 1 );
      if( ! ref_imgs[i] )
	{
	  fprintf(stderr, "Error: could not load reference image from %s\n",
		  argv[optind]);
	  num_ref_imgs--;
	}
      else
	i++;
      optind++;
    }
}
