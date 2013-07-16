/*
  Perform single object tracking with particle filtering

  @author Rob Hess
  @version 1.0.0-20060306
*/

#include "defs.h"
#include "utils.h"
#include "particles.h"
#include "observation.h"
#include <ctime>

/******************************** Definitions ********************************/

/* command line options */
#define OPTIONS ":p:oah"

/* default number of particles */
#define PARTICLES 100

/* default basename and extension of exported frames */
#define EXPORT_BASE "./frames/frame_"
#define EXPORT_EXTN ".png"

/* maximum number of frames for exporting */
#define MAX_FRAMES 2048

/********************************* Structures ********************************/

/* maximum number of objects to be tracked */
#define MAX_OBJECTS 1

typedef struct params {
  CvPoint loc1[MAX_OBJECTS];
  CvPoint loc2[MAX_OBJECTS];
  IplImage* objects[MAX_OBJECTS];
  char* win_name;
  IplImage* orig_img;
  IplImage* cur_img;
  int n;
} params;


/***************************** Function Prototypes ***************************/

void usage( char* );
void arg_parse( int, char** );
int get_regions( IplImage*, CvRect** );
void mouse( int, int, int, int, void* );
histogram** compute_ref_histos( IplImage*, CvRect*, int );
int export_ref_histos( histogram**, int );
int export_frame( IplImage*, int );


/********************************** Globals **********************************/

char* pname;                      /* program name */
char* vid_file;                   /* input video file name */
int num_particles = PARTICLES;    /* number of particles */
int show_all = FALSE;             /* TRUE to display all particles */
int export = FALSE;               /* TRUE to exported tracking sequence */


/*********************************** Main ************************************/

int main( int argc, char** argv )
{
  gsl_rng* rng;
  IplImage* frame, * hsv_frame, * frames[MAX_FRAMES];
  IplImage** hsv_ref_imgs;
  histogram** ref_histos;
  CvCapture* video;
  particle* particles, * new_particles;
  CvScalar color;
  CvRect* regions;
  int num_objects = 0;
  float s;
  int i, j, k, w, h, x, y;

  /* parse command line and initialize random number generator */
  arg_parse( argc, argv );
  gsl_rng_env_setup();
  rng = gsl_rng_alloc( gsl_rng_mt19937 );
  gsl_rng_set( rng, std::time(NULL) );

  video = cvCaptureFromFile( vid_file );
  if( ! video )
    fatal_error("couldn't open video file %s", vid_file);
    
  i = 0;
  while( frame = cvQueryFrame( video ) )
    {
      hsv_frame = bgr2hsv( frame );
      frames[i] = cvClone( frame );

      /* allow user to select object to be tracked in the first frame */
      if( i == 0 )
	{
	  w = frame->width;
	  h = frame->height;
	  fprintf( stderr, "Select object region to track\n" );
	  while( num_objects == 0 )
	    {
	      num_objects = get_regions( frame, &regions );
	      if( num_objects == 0 )
		fprintf( stderr, "Please select a object\n" );
	    }

	  /* compute reference histograms and distribute particles */
	  ref_histos = compute_ref_histos( hsv_frame, regions, num_objects );
	  if( export )
	    export_ref_histos( ref_histos, num_objects );
	  particles = init_distribution( regions, ref_histos,
					 num_objects, num_particles );
	}
      else
	{
	  /* perform prediction and measurement for each particle */
	  for( j = 0; j < num_particles; j++ )
	    {
	      particles[j] = transition( particles[j], w, h, rng );
	      s = particles[j].s;
	      particles[j].w = likelihood( hsv_frame, cvRound(particles[j].y),
					   cvRound( particles[j].x ),
					   cvRound( particles[j].width * s ),
					   cvRound( particles[j].height * s ),
					   particles[j].histo );
	    }
	  
	  /* normalize weights and resample a set of unweighted particles */
	  normalize_weights( particles, num_particles );
	  new_particles = resample( particles, num_particles );
	  free( particles );
	  particles = new_particles;
	}

      /* display all particles if requested */
      qsort( particles, num_particles, sizeof( particle ), &particle_cmp );
      if( show_all )
	for( j = num_particles - 1; j > 0; j-- )
	  {
	    color = CV_RGB(0,0,255);
	    display_particle( frames[i], particles[j], color );
	  }
      
      /* display most likely particle */
      color = CV_RGB(255,0,0);
      display_particle( frames[i], particles[0], color );
      cvNamedWindow( "Video", 1 );
      cvShowImage( "Video", frames[i] );
      cvWaitKey( 5 );
    
      cvReleaseImage( &hsv_frame );
      i++;
    }
  cvReleaseCapture( &video );
  /* export video frames, if export requested */
  if( export )
    fprintf( stderr, "Exporting video frames... " );
  for( j = 0; j < i; j++ )
    {
      if( export )
	{
	  progress( FALSE );
	  export_frame( frames[j], j+1 );
	}
      cvReleaseImage( &frames[j] );
    }
  if( export )
    progress( TRUE );
}


/************************** Function Definitions *****************************/

/* print usage for this program */
void usage( char* name )
{
  fprintf(stderr, "%s: track a single object using particle filtering\n\n",
	  name);
  fprintf(stderr, "Usage: %s [options] <vid_file>\n\n", name);
  fprintf(stderr, "Arguments:\n");
  fprintf(stderr, "  <vid_file>          A clip of video in which " \
	  "to track an object\n");
  fprintf(stderr, "\nOptions:\n");
  fprintf(stderr, "  -h                  Display this message and exit\n");
  fprintf(stderr, "  -a                  Display all particles, not just " \
	  "the most likely\n");
  fprintf(stderr, "  -o                  Output tracking sequence frames as " \
	  "%s*%s\n", EXPORT_BASE, EXPORT_EXTN);
  fprintf(stderr, "  -p <particles>      Number of particles (default %d)\n",
	  PARTICLES);
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
	  
	case 'a':
	  show_all = TRUE;
	  break;

	  /* user wants to output tracking sequence */
	case 'o':
	  export = TRUE;
	  break;

	  /* user wants to set number of particles */
	case 'p':
	  if( ! optarg )
	    fatal_error( "error parsing arguments at -%c\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  num_particles = strtol( optarg, &arg_check, 10 );
	  if( arg_check == optarg  ||  *arg_check != '\0' )
	    fatal_error( "-%c option requires an integer argument\n"	\
			 "Try '%s -h' for help.", arg, pname );
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
  if( argc - optind > 2 )
    fatal_error( "too many arguments.\nTry '%s -h' for help.", pname );
 
  /* record video file name */
  vid_file = argv[optind];
}



/*
  Allows the user to interactively select object regions.

  @param frame the frame of video in which objects are to be selected
  @param regions a pointer to an array to be filled with rectangles
    defining object regions

  @return Returns the number of objects selected by the user
*/
int get_regions( IplImage* frame, CvRect** regions )
{
  char* win_name = "First frame";
  params p;
  CvRect* r;
  int i, x1, y1, x2, y2, w, h;
  
  /* use mouse callback to allow user to define object regions */
  p.win_name = win_name;
  p.orig_img = cvClone( frame );
  p.cur_img = NULL;
  p.n = 0;
  cvNamedWindow( win_name, 1 );
  cvShowImage( win_name, frame );
  cvSetMouseCallback( win_name, &mouse, &p );
  cvWaitKey( 0 );
  cvDestroyWindow( win_name );
  cvReleaseImage( &(p.orig_img) );
  if( p.cur_img )
    cvReleaseImage( &(p.cur_img) );

  /* extract regions defined by user; store as an array of rectangles */
  if( p.n == 0 )
    {
      *regions = NULL;
      return 0;
    }
  r = malloc( p.n * sizeof( CvRect ) );
  for( i = 0; i < p.n; i++ )
    {
      x1 = MIN( p.loc1[i].x, p.loc2[i].x );
      x2 = MAX( p.loc1[i].x, p.loc2[i].x );
      y1 = MIN( p.loc1[i].y, p.loc2[i].y );
      y2 = MAX( p.loc1[i].y, p.loc2[i].y );
      w = x2 - x1;
      h = y2 - y1;

      /* ensure odd width and height */
      w = ( w % 2 )? w : w+1;
      h = ( h % 2 )? h : h+1;
      r[i] = cvRect( x1, y1, w, h );
    }
  *regions = r;
  return p.n;
}



/*
  Mouse callback function that allows user to specify the initial object
  regions.  Parameters are as specified in OpenCV documentation.
*/
void mouse( int event, int x, int y, int flags, void* param )
{
  params* p = (params*)param;
  CvPoint* loc;
  int n;
  IplImage* tmp;
  static int pressed = FALSE;
  
  /* on left button press, remember first corner of rectangle around object */
  if( event == CV_EVENT_LBUTTONDOWN )
    {
      n = p->n;
      if( n == MAX_OBJECTS )
	return;
      loc = p->loc1;
      loc[n].x = x;
      loc[n].y = y;
      pressed = TRUE;
    }

  /* on left button up, finalize the rectangle and draw it in black */
  else if( event == CV_EVENT_LBUTTONUP )
    {
      n = p->n;
      if( n == MAX_OBJECTS )
	return;
      loc = p->loc2;
      loc[n].x = x;
      loc[n].y = y;
      cvReleaseImage( &(p->cur_img) );
      p->cur_img = NULL;
      cvRectangle( p->orig_img, p->loc1[n], loc[n], CV_RGB(0,0,0), 1, 8, 0 );
      cvShowImage( p->win_name, p->orig_img );
      pressed = FALSE;
      p->n++;
    }

  /* on mouse move with left button down, draw rectangle as defined in white */
  else if( event == CV_EVENT_MOUSEMOVE  &&  flags & CV_EVENT_FLAG_LBUTTON )
    {
      n = p->n;
      if( n == MAX_OBJECTS )
	return;
      tmp = cvClone( p->orig_img );
      loc = p->loc1;
      cvRectangle( tmp, loc[n], cvPoint(x, y), CV_RGB(255,255,255), 1, 8, 0 );
      cvShowImage( p->win_name, tmp );
      if( p->cur_img )
	cvReleaseImage( &(p->cur_img) );
      p->cur_img = tmp;
    }
}



/*
  Computes a reference histogram for each of the object regions defined by
  the user

  @param frame video frame in which to compute histograms; should have been
    converted to hsv using bgr2hsv in observation.h
  @param regions regions of \a frame over which histograms should be computed
  @param n number of regions in \a regions
  @param export if TRUE, object region images are exported

  @return Returns an \a n element array of normalized histograms corresponding
    to regions of \a frame specified in \a regions.
*/
histogram** compute_ref_histos( IplImage* frame, CvRect* regions, int n )
{
  histogram** histos = malloc( n * sizeof( histogram* ) );
  IplImage* tmp;
  int i;

  /* extract each region from frame and compute its histogram */
  for( i = 0; i < n; i++ )
    {
      cvSetImageROI( frame, regions[i] );
      tmp = cvCreateImage( cvGetSize( frame ), IPL_DEPTH_32F, 3 );
      cvCopy( frame, tmp, NULL );
      cvResetImageROI( frame );
      histos[i] = calc_histogram( &tmp, 1 );
      normalize_histogram( histos[i] );
      cvReleaseImage( &tmp );
    }

  return histos;
}



/*
  Exports reference histograms to file

  @param ref_histos array of reference histograms
  @param n number of histograms

  @return Returns 1 on success or 0 on failure
*/
int export_ref_histos( histogram** ref_histos, int n )
{
  char name[32];
  char num[3];
  FILE* file;
  int i;
  
  for( i = 0; i < n; i++ )
    {
      snprintf( num, 3, "%02d", i );
      strcpy( name, "hist_" );
      strcat( name, num );
      strcat( name, ".dat" );
      if( ! export_histogram( ref_histos[i], name ) )
	return 0;
    }

  return 1;
}



/*
  Exports a frame whose name and format are determined by EXPORT_BASE and
  EXPORT_EXTN, defined above.

  @param frame frame to be exported
  @param i frame number
*/
int export_frame( IplImage* frame, int i )
{
  char name[ strlen(EXPORT_BASE) + strlen(EXPORT_EXTN) + 4 ];
  char num[5];

  snprintf( num, 5, "%04d", i );
  strcpy( name, EXPORT_BASE );
  strcat( name, num );
  strcat( name, EXPORT_EXTN );
  return cvSaveImage( name, frame );
}
