#include <stdio.h>
#include <math.h>

#ifdef ANSI_TERM_SEQUENCES
#include <unistd.h>
#endif

#include "data_types.h"
#include "model_parameters.h"
#include "condensation.h"

/* The following routines are model-specific and should be replaced to
   implement an arbitrary process and observation model. */

void initialise_model_specific_defaults()
{
  /* Set up the parameters of the simulation model, process and
     observation. */
  global.scene.process.mean = SimulatedMean;
  global.scene.process.scaling = SimulatedScaling;
  global.scene.process.sigma = SimulatedSigma;
  global.scene.sigma = SimulatedMeasSigma;

  /* Set up the parameters of the prior distribution */
  global.prior.mean = PriorMean;
  global.prior.sigma = PriorSigma;

  /* Set up the parameters of the process model */
  global.process.mean = ProcessMean;
  global.process.scaling = ProcessScaling;
  global.process.sigma = ProcessSigma;

  /* Set up the parameters of the observation model */
  global.obs.sigma = ObsSigma;

  /* Set up the parameters of the display model */
  global.disp.histogram_width = DisplayWidth;
}

/* Set up the initial sample-set according to the prior model. The
   prior is a simple Gaussian, so each of the positions is filled in
   by sampling that Gaussian and the weights are initialised to be
   uniform. gaussian_random can be found in utility.c */
void set_up_prior_conditions()
{
  int n;

  for (n=0; n<global.nsamples; ++n) {
    data.old_positions[n] =
      global.prior.mean + global.prior.sigma * gaussian_random();

    /* The probabilities are not normalised. */
    data.cumul_prob_array[n] = (double) n;
    data.sample_weights[n] = 1.0;
  }

  /* The probabilities are not normalised, so store the largest value
     here (for simplicity of the binary search algorithm,
     cumul_prob_array[0] = 0). This can then be used as a
     multiplicative normalisation constant for the sample_weights
     array, as well. */
  data.largest_cumulative_prob = (double) n;

  /* This is the initial positions the simulated object. */
  //--S [] 2013/07/16: Sang-Wook Lee
  //data.meas.true = 0.0;
  data.meas.truth = 0.0;
  //--E [] 2013/07/16: Sang-Wook Lee
}

/* The process model for a first-order auto-regressive process is:

   x_{t+1} - mean = (x_t - mean)*scaling + sigma*w_t

   where w_t is unit iid Gaussian noise. gaussian_random can be found
   in utility.c */
float iterate_first_order_arp(float previous, ProcessModel process)
{
  return process.mean +
    ((previous - process.mean) * process.scaling) +
    process.sigma * gaussian_random();
}

/* In a real implementation, this routine would go and actually make
   measurements and store them in the data.meas structure. This
   simulation consists of an `object' moving around obeying a
   first-order auto-regressive process, and being observed with its
   true positions corrupted by Gaussian measurement noise.
   Accordingly, this routine calculates the new simulated true and
   measured position of the object. */
void obtain_observations()
{
  //--S [] 2013/07/16: Sang-Wook Lee
/*
  data.meas.true =
    iterate_first_order_arp(data.meas.true,
			    global.scene.process);

  data.meas.observed =
    data.meas.true +
    global.scene.sigma * gaussian_random();
*/
  data.meas.truth =
    iterate_first_order_arp(data.meas.truth,
			    global.scene.process);

  data.meas.observed =
    data.meas.truth +
    global.scene.sigma * gaussian_random();
  //--E [] 2013/07/16: Sang-Wook Lee
}

/* This routine samples from the distribution

   p(x_t | x_{t-1} = old_positions[old_sample])

   and stores the result in new_positions[new_sample]. This is
   straightforward for the simple first-order auto-regressive process
   model used here, but any model could be substituted. */
void predict_sample_position(int new_sample, int old_sample)
{
  data.new_positions[new_sample] =
    iterate_first_order_arp(data.old_positions[old_sample], global.process);
}

/* This routine evaluates the observation density

   p(z_t|x_t = new_positions[new_sample])

   The observation model in this implementation is a simple mixture of
   Gaussians, where each simulated object is observed as a 1d position
   and measurement noise is represented as Gaussian. For a
   visual-tracking application, this routine would go and evaluate the
   likelihood that the object is present in the image at the position
   encoded by new_positions[new_sample]. evaluate_gaussian can be
   found in utility.c */
double evaluate_observation_density(int new_sample)
{
  return
    evaluate_gaussian(data.new_positions[new_sample] -
		      data.meas.observed, global.obs.sigma);
}

/* The following two routines provide rudimentary graphical output in
   the form of an ASCII histogram of the 1d state density. */
static int eval_bin(float where)
{
  return (HistBins-1)/2 +
    (int) (where * (HistBins-1)/2 / global.disp.histogram_width);
}

static void display_histogram(double estimated_position)
{
  static double bins[HistBins];

  int b, n, line, which_bin, meas_bin, est_bin, true_bin;
  double lineheight;
  char outc;

  for (b=0; b<HistBins; ++b)
    bins[b] = 0.0;

  for (n=0; n<global.nsamples; ++n) {
    which_bin = eval_bin(data.new_positions[n]);

    if (which_bin >=0 && which_bin < HistBins)
      bins[which_bin] +=
	data.sample_weights[n] / data.largest_cumulative_prob;
  }

  for (b=0; b<HistBins; ++b)
    bins[b] = (bins[b] * (double) HistLines) / MaxHistHeight;

  for (line=0; line<HistLines; ++line) {
    lineheight = (double) (HistLines-1-line);
    for (b=0; b<HistBins; ++b) {
      if (line==0 && bins[b] >= lineheight+1.0)
	outc = '*';
      else if (bins[b] >= lineheight+0.5 && bins[b] < lineheight+1.0)
	outc = '-';
      else if (bins[b] >= lineheight && bins[b] < lineheight+0.5)
	outc = '_';
      else
	outc = ' ';
      printf("%c", outc);
    }
    printf("\n");
  }

  //--S [] 2013/07/16: Sang-Wook Lee
  //true_bin = eval_bin(data.meas.true);
  true_bin = eval_bin(data.meas.truth);
  //--E [] 2013/07/16: Sang-Wook Lee
  meas_bin = eval_bin(data.meas.observed);
  est_bin = eval_bin(estimated_position);

  for (b=0; b<HistBins; ++b) {
    if ((b == meas_bin && b == est_bin) ||
	(b == meas_bin && b == true_bin) ||
	(b == true_bin && b == est_bin))
      outc = '*';
    else if (b == true_bin)
      outc = '.';
    else if (b == meas_bin)
      outc = '+';
    else if (b == est_bin)
      outc = 'x';
    else
      outc = ' ';
    printf("%c", outc);
  }
  printf("\n");
}

/* This routine computes the estimated position of the object by
   estimating the mean of the state-distribution as a weighted mean of
   the sample positions, then displays a histogram of the state
   distribution along with a character denoting the estimated position
   of the object. */
void display_data(int iteration)
{
  int n;
  double aggregate;

  aggregate = 0.0;

  /* Compute the unnormalised weighted mean of the sample
     positions. */
  for (n=0; n<global.nsamples; ++n)
    aggregate += data.new_positions[n] * data.sample_weights[n];

  aggregate /= data.largest_cumulative_prob;

  display_histogram(aggregate);

  printf("%04d: Measured pos. % 3.4lf True pos. % 3.4lf Est. position % 3.4lf\n",
	 //--S [] 2013/07/16: Sang-Wook Lee
	 //iteration, data.meas.observed, data.meas.true, aggregate);
	 iteration, data.meas.observed, data.meas.truth, aggregate);
	 //--E [] 2013/07/16: Sang-Wook Lee

#ifdef ANSI_TERM_SEQUENCES
  /* If possible, run the cursor back up the screen so the histogram
     stays in the same place instead of scrolling down the display. */
  printf("\033[%dA", HistLines+2);
  sleep(1);
#endif
}

/* End of model-specific routines */
