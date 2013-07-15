/* The following are model-specific definitions */

/* The state vector for this simple model is one-dimensional. In
   multiple dimensions, for example, Sample would be an array. */
typedef double Sample;

/* The process model used in this simple implementation is a
   one-dimensional first-order auto-regressive process, where

   (x_t - mean) = (x_{t-1} - mean) * scaling + sigma w_t

   and the w_t are iid unit zero-mean Gaussian noise samples. */
typedef struct _ProcessModel {
  double mean, scaling, sigma;
} ProcessModel;

/* This structure contains the parameters of the simulation model. The
   simulated data is produced from a model of a single particle
   following a one-dimensional 1st-order ARP whose parameters are
   stored in the structure process. Measurements are simulated by
   adding Gaussian noise with std. deviation sigma to the true
   simulated position of the particle. */
typedef struct _SceneModel {
  ProcessModel process;
  double sigma;
} SceneModel;

/* This structure contains the data for the measurements made at each
   timestep. The simulation consists of a single point-measurement.
   This structure stores the true position of the simulated particle,
   and its measured position, which is corrupted by noise. */
typedef struct _MeasData {
  //--S [] 2013/07/16: Sang-Wook Lee
  //double true, observed;
  double truth, observed;
  //--E [] 2013/07/16: Sang-Wook Lee
} MeasData;

/* The prior distribution of the state is taken to be Gaussian with
   the parameters stored in this structure. */
typedef struct _PriorModel {
  double mean, sigma;
} PriorModel;

/* The observation model is of Gaussian noise with std. deviation
   sigma, so

   z_t = x_t + sigma v_t

   where the v_t are assumed iid unit zero-mean Gaussian noise
   samples. In principle for this simulation, the modelled observation
   noise can be different to the simulated observation noise, hence
   the presence of a separate std. deviation parameter in the
   structure SceneModel. */
typedef struct _ObservationModel {
  double sigma;
} ObservationModel;

/* This structure contains information about how the state
   distribution should be displayed at each timestep. The display in
   this implementation is a one-dimensional histogram, and this
   parameter sets the width of the histogram (so that the interval
   [-histogram_width, histogram_width] is displayed). */
typedef struct _DisplayModel {
  double histogram_width;
} DisplayModel;

/* End of model-specific structures */


/* The following should work with any model-specific definitions */

/* This structure contains all the parameter settings for the models
   which remain constant throughout a run of the algorithm. */
typedef struct _GlobalData {
  PriorModel prior;     /* The parameters specifying the model of the
			   prior distribution for the first
			   timestep. */
  ProcessModel process; /* The parameters specifying the process
			   model. */
  SceneModel scene;     /* The parameters specifying the simulation
			   model of the scene. This is only used in
			   the case of simulated data. */
  ObservationModel obs; /* The parameters specifying the observation
			   model. */
  DisplayModel disp;    /* The parameters specifying how to display
			   the state estimate at each timestep. */
  int nsamples;         /* The number of samples N. */
  int niterations;      /* The number of iterations to run the
			   filter. */
} GlobalData;

/* This structure contains all of the information which is specific to
   a given iteration of the algorithm. */
typedef struct _IterationData {
  /* The following arrays contain the sample positions for the current
     and previous timesteps respectively. At the end of each
     iteration, these pointers are swapped over to avoid copying data
     structures, so their addresses should not be relied on. */
  Sample *new_positions, *old_positions;

  /* The following arrays give the sample weights and cumulative
     probabilities, as well as the largest cumulative
     probability. There is no stage in the algorithm when the weights
     from both the previous and current timesteps are needed. At the
     beginning of an iteration, sample_weights contains the weights
     from the previous iteration, and by the end it contains the
     weights of the current iteration. The cumulative probabilities
     are not normalised, so largest_cumulative_prob is needed to store
     the largest cumulative probability (for simplicity of the binary
     search algorithm, cumul_prob_array[0] = 0) */
  double *sample_weights, *cumul_prob_array, largest_cumulative_prob;

  /* The measurements made in a given iteration are stored here. For
     some applications a discrete set of measurements is not
     appropriate, and this could contain, e.g. a pointer to an image
     structure. */
  MeasData meas;
} IterationData;

/* End of generic structures */


/* All of the global information is packaged into the following two
   structures. `global' contains information which is constant over a
   run of the algorithm and `data' contains all of the current state
   at a given iteration. */

extern GlobalData global;
extern IterationData data;

/* End of global variables */
