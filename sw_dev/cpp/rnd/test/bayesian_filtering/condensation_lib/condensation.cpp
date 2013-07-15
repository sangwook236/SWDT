#include <stdlib.h>
#include <stdio.h>

#include "data_types.h"
#include "model_parameters.h"
#include "condensation.h"

/* All of the global information is packaged into the following two
   structures. `global' contains information which is constant over a
   run of the algorithm and `data' contains all of the current state
   at a given iteration. */

GlobalData global;
IterationData data;

/* End of global variables */


/* From here on is generic Condensation regardless of the form of
   model or observation used. All of the model-specific routines are
   found in model_specific.c */

/* This is binary search using cumulative probabilities to pick a base
   sample. The use of this routine makes Condensation O(NlogN) where N
   is the number of samples. It is probably better to pick base
   samples deterministically, since then the algorithm is O(N) and
   probably marginally more efficient, but this routine is kept here
   for conceptual simplicity and because it maps better to the
   published literature. */
int pick_base_sample(void)
{
  double choice = uniform_random() * data.largest_cumulative_prob;
  int low, middle, high;

  low = 0;
  high = global.nsamples;

  while (high>(low+1)) {
    middle = (high+low)/2;
    if (choice > data.cumul_prob_array[middle])
      low = middle;
    else high = middle;
  }

  return low;
}

/* This routine computes all of the new (unweighted) sample
   positions. For each sample, first a base is chosen, then the new
   sample position is computed by sampling from the prediction density
   p(x_t|x_t-1 = base). predict_sample_position is obviously
   model-dependent and is found in model_specific.c, but it can be
   replaced by any process model required. */
void predict_new_bases(void)
{
  int n, base;

  for (n=0; n<global.nsamples; ++n) {
    base = pick_base_sample();
    predict_sample_position(n, base);
  }
}

/* Once all the unweighted sample positions have been computed using
   predict_new_bases, this routine computes the weights by evaluating
   the observation density at each of the positions. Cumulative
   probabilities are also computed at the same time, to permit an
   efficient implementation of pick_base_sample using binary
   search. evaluate_observation_density is obviously model-dependent
   and is found in model_specific.c, but it can be replaced by any
   observation model required. */
void calculate_base_weights(void)
{
  int n;
  double cumul_total;

  cumul_total = 0.0;
  for (n=0; n<global.nsamples; ++n) {
    data.sample_weights[n] = evaluate_observation_density(n);
    data.cumul_prob_array[n] = cumul_total;
    cumul_total += data.sample_weights[n];
  }
  data.largest_cumulative_prob = cumul_total;
}

/* Go and output the estimate for this iteration (which is a
   model-dependent routine found in model_specific.c) and then swap
   over the arrays ready for the next iteration. */
void update_after_iterating(int iteration)
{
  Sample *temp;

  display_data(iteration);

  temp = data.new_positions;
  data.new_positions = data.old_positions;
  data.old_positions = temp;
}

/* obtain_observations is model-dependent and can be found in
   model_specific.c */
void run_filter(void)
{
  int i;

  for (i=0; i<global.niterations; ++i) {
    obtain_observations();     /* Go make necessary measurements */
    predict_new_bases();       /* Push previous state through process model */
    calculate_base_weights();  /* Apply Bayesian measurement weighting */
    update_after_iterating(i); /* Tidy up, display output, etc. */
  }
}

/* This routine fills in the data structures with default constant
   values. It could be enhanced by reading informatino from the
   command line to allow e.g. N to be altered without recompiling. */
void initialise_defaults(void)
{
  global.nsamples = NSamples;
  global.niterations = NIterations;

  initialise_model_specific_defaults();
}

/* Create all the arrays, then fill in the prior distribution for the
   first iteration. The prior is model-dependent, so
   set_up_prior_conditions can be found in model_specific.c */
int initialise(int argc, char *argv[])
{
  initialise_defaults();

  data.new_positions = malloc(sizeof(Sample) * global.nsamples);
  data.old_positions = malloc(sizeof(Sample) * global.nsamples);

  data.sample_weights = malloc(sizeof(double) * global.nsamples);
  data.cumul_prob_array = malloc(sizeof(double) * global.nsamples);

  if (!data.new_positions || !data.old_positions ||
      !data.sample_weights || !data.cumul_prob_array) {
    fprintf(stderr, "Failed to allocate memory for sample arrays\n");
    return 0;
  }

  set_up_prior_conditions();

  return 1;
}

/* Tidy up */
void shut_down(void)
{
  free(data.new_positions);
  free(data.old_positions);
  free(data.sample_weights);
  free(data.cumul_prob_array);
}

int main(int argc, char *argv[])
{
  if (!initialise(argc, argv))
    return 1;

  run_filter();

  shut_down();

  return 0;
}
