/*
  This file contains general program definitions.
  
  @author Rob Hess
  @version 1.0.0-20060306
*/

#ifndef DEFS_H
#define DEFS_H

/********************************* Includes **********************************/

/* From standard C library */
#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
#include <unistd.h>
#endif

/* From OpenCV library */
//--S [] 2013/07/16: Sang-Wook Lee
/*
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
*/
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>
#include <opencv2/legacy/legacy.hpp>
//--E [] 2013/07/16: Sang-Wook Lee

/* From GSL */
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


/******************************* Defs and macros *****************************/

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef MIN
#define MIN(x,y) ( ( x < y )? x : y )
#endif
#ifndef MAX
#define MAX(x,y) ( ( x > y )? x : y )
#endif
#ifndef ABS
#define ABS(x) ( ( x < 0 )? -x : x )
#endif

/********************************** Structures *******************************/

#endif
