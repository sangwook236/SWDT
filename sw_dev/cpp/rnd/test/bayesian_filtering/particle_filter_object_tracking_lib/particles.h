/** @file
    Definitions related to tracking with particle filtering

    @author Rob Hess
    @version 1.0.0-20060307
*/

#ifndef PARTICLES_H
#define PARTICLES_H

#include "observation.h"

/******************************* Definitions *********************************/

/* standard deviations for gaussian sampling in transition model */
#define TRANS_X_STD 1.0
#define TRANS_Y_STD 0.5
#define TRANS_S_STD 0.001

/* autoregressive dynamics parameters for transition model */
#define A1  2.0
#define A2 -1.0
#define B0  1.0000

/******************************* Structures **********************************/

/**
   A particle is an instantiation of the state variables of the system
   being monitored.  A collection of particles is essentially a
   discretization of the posterior probability of the system.
*/
typedef struct particle {
  float x;          /**< current x coordinate */
  float y;          /**< current y coordinate */
  float s;          /**< scale */
  float xp;         /**< previous x coordinate */
  float yp;         /**< previous y coordinate */
  float sp;         /**< previous scale */
  float x0;         /**< original x coordinate */
  float y0;         /**< original y coordinate */
  int width;        /**< original width of region described by particle */
  int height;       /**< original height of region described by particle */
  histogram* histo; /**< reference histogram describing region being tracked */
  float w;          /**< weight */
} particle;


/**************************** Function Prototypes ****************************/

/**
   Creates an initial distribution of particles by sampling from a Gaussian
   window around each of a set of specified locations
   
   
   @param regions an array of regions describing player locations around
     which particles are to be sampled
   @param histos array of histograms describing regions in \a regions
   @param n the number of regions in \a regions
   @param p the total number of particles to be assigned
   
   @return Returns an array of \a p particles sampled from around regions in
     \a regions
*/
particle* init_distribution(CvRect* regions, histogram** histos, int n, int p);


/**
   Samples a transition model for a given particle

   @param p a particle to be transitioned
   @param w video frame width
   @param h video frame height
   @param rng a random number generator from which to sample

   @return Returns a new particle sampled based on <EM>p</EM>'s transition
     model
*/
particle transition( particle p, int w, int h, gsl_rng* rng );


/**
   Normalizes particle weights so they sum to 1

   @param particles an array of particles whose weights are to be normalized
   @param n the number of particles in \a particles
*/
void normalize_weights( particle* particles, int n );


/**
   Re-samples a set of weighted particles to produce a new set of unweighted
   particles

   @param particles an old set of weighted particles whose weights have been
     normalized with normalize_weights()
   @param n the number of particles in \a particles
  
   @return Returns a new set of unweighted particles sampled from \a particles
*/
particle* resample( particle* particles, int n );


/**
   Compare two particles based on weight.  For use in qsort.

   @param p1 pointer to a particle
   @param p2 pointer to a particle

   @return Returns -1 if the \a p1 has lower weight than \a p2, 1 if \a p1
     has higher weight than \a p2, and 0 if their weights are equal.
*/
int particle_cmp( const void* p1, const void* p2 );


/**
   Displays a particle on an image as a rectangle around the region specified
   by the particle

   @param img the image on which to display the particle
   @param p the particle to be displayed
   @param color the color in which \a p is to be displayed
*/
void display_particle( IplImage* img, particle p, CvScalar color );


#endif
