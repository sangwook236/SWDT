/*
  Functions for object tracking with a particle filter
  
  @author Rob Hess
  @version 1.0.0-20060310
*/


#include "defs.h"
#include "utils.h"
#include "particles.h"



/*************************** Function Definitions ****************************/

/*
  Creates an initial distribution of particles at specified locations
  
  
  @param regions an array of regions describing player locations around
    which particles are to be sampled
  @param histos array of histograms describing regions in \a regions
  @param n the number of regions in \a regions
  @param p the total number of particles to be assigned
  
  @return Returns an array of \a p particles sampled from around regions in
    \a regions
*/
particle* init_distribution( CvRect* regions, histogram** histos, int n, int p)
{
  particle* particles;
  int np;
  float x, y;
  int i, j, width, height, k = 0;
  
  particles = (particle *)malloc( p * sizeof( particle ) );
  np = p / n;

  /* create particles at the centers of each of n regions */
  for( i = 0; i < n; i++ )
    {
      width = regions[i].width;
      height = regions[i].height;
      x = regions[i].x + width / 2;
      y = regions[i].y + height / 2;
      for( j = 0; j < np; j++ )
	{
	  particles[k].x0 = particles[k].xp = particles[k].x = x;
	  particles[k].y0 = particles[k].yp = particles[k].y = y;
	  particles[k].sp = particles[k].s = 1.0;
	  particles[k].width = width;
	  particles[k].height = height;
	  particles[k].histo = histos[i];
	  particles[k++].w = 0;
	}
    }

  /* make sure to create exactly p particles */
  i = 0;
  while( k < p )
    {
      width = regions[i].width;
      height = regions[i].height;
      x = regions[i].x + width / 2;
      y = regions[i].y + height / 2;
      particles[k].x0 = particles[k].xp = particles[k].x = x;
      particles[k].y0 = particles[k].yp = particles[k].y = y;
      particles[k].sp = particles[k].s = 1.0;
      particles[k].width = width;
      particles[k].height = height;
      particles[k].histo = histos[i];
      particles[k++].w = 0;
      i = ( i + 1 ) % n;
    }

  return particles;
}



/*
  Samples a transition model for a given particle
  
  @param p a particle to be transitioned
  @param w video frame width
  @param h video frame height
  @param rng a random number generator from which to sample

  @return Returns a new particle sampled based on <EM>p</EM>'s transition
    model
*/
particle transition( particle p, int w, int h, gsl_rng* rng )
{
  float x, y, s;
  particle pn;
  
  /* sample new state using second-order autoregressive dynamics */
  x = A1 * ( p.x - p.x0 ) + A2 * ( p.xp - p.x0 ) +
    B0 * gsl_ran_gaussian( rng, TRANS_X_STD ) + p.x0;
  pn.x = MAX( 0.0, MIN( (float)w - 1.0, x ) );
  y = A1 * ( p.y - p.y0 ) + A2 * ( p.yp - p.y0 ) +
    B0 * gsl_ran_gaussian( rng, TRANS_Y_STD ) + p.y0;
  pn.y = MAX( 0.0, MIN( (float)h - 1.0, y ) );
  s = A1 * ( p.s - 1.0 ) + A2 * ( p.sp - 1.0 ) +
    B0 * gsl_ran_gaussian( rng, TRANS_S_STD ) + 1.0;
  pn.s = MAX( 0.1, s );
  pn.xp = p.x;
  pn.yp = p.y;
  pn.sp = p.s;
  pn.x0 = p.x0;
  pn.y0 = p.y0;
  pn.width = p.width;
  pn.height = p.height;
  pn.histo = p.histo;
  pn.w = 0;

  return pn;
}



/*
  Normalizes particle weights so they sum to 1
  
  @param particles an array of particles whose weights are to be normalized
  @param n the number of particles in \a particles
*/
void normalize_weights( particle* particles, int n )
{
  float sum = 0;
  int i;

  for( i = 0; i < n; i++ )
    sum += particles[i].w;
  for( i = 0; i < n; i++ )
    particles[i].w /= sum;
}



/*
  Re-samples a set of particles according to their weights to produce a
  new set of unweighted particles
  
  @param particles an old set of weighted particles whose weights have been
    normalized with normalize_weights()
  @param n the number of particles in \a particles
  
  @return Returns a new set of unweighted particles sampled from \a particles
*/
particle* resample( particle* particles, int n )
{
  particle* new_particles;
  int i, j, np, k = 0;

  qsort( particles, n, sizeof( particle ), &particle_cmp );
  new_particles = (particle *)malloc( n * sizeof( particle ) );
  for( i = 0; i < n; i++ )
    {
      np = cvRound( particles[i].w * n );
      for( j = 0; j < np; j++ )
	{
	  new_particles[k++] = particles[i];
	  if( k == n )
	    goto exit;
	}
    }
  while( k < n )
    new_particles[k++] = particles[0];

 exit:
  return new_particles;
}



/*
  Compare two particles based on weight.  For use in qsort.

  @param p1 pointer to a particle
  @param p2 pointer to a particle

  @return Returns -1 if the \a p1 has lower weight than \a p2, 1 if \a p1
    has higher weight than \a p2, and 0 if their weights are equal.
*/
int particle_cmp( const void* p1, const void* p2 )
{
  particle* _p1 = (particle*)p1;
  particle* _p2 = (particle*)p2;

  if( _p1->w > _p2->w )
    return -1;
  if( _p1->w < _p2->w )
    return 1;
  return 0;
}



/*
  Displays a particle on an image as a rectangle around the region specified
  by the particle
  
  @param img the image on which to display the particle
  @param p the particle to be displayed
  @param color the color in which \a p is to be displayed
*/
void display_particle( IplImage* img, particle p, CvScalar color )
{
  int x0, y0, x1, y1;

  x0 = cvRound( p.x - 0.5 * p.s * p.width );
  y0 = cvRound( p.y - 0.5 * p.s * p.height );
  x1 = x0 + cvRound( p.s * p.width );
  y1 = y0 + cvRound( p.s * p.height );
  
  cvRectangle( img, cvPoint( x0, y0 ), cvPoint( x1, y1 ), color, 1, 8, 0 );
}
