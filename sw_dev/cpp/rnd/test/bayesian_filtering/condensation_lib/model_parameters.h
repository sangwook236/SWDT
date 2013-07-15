/* The following are all the constants controlling the behaviour of
   the system and output. */

/* How many samples in the distribution? */
#define NSamples (1000)

/* How many iterations to run the filter? */
#define NIterations (100)

/* The simulated object follows a model of the same form as the
   process */
#define SimulatedMean (-0.1)
#define SimulatedScaling (0.4)
#define SimulatedSigma (0.075)
#define SimulatedMeasSigma (0.03)

/* The prior distribution over samples at the first timestep is a
   Gaussian with the following parameters. */
#define PriorMean (0.0)
#define PriorSigma (0.2)

/* The process model is a first-order Auto-Regressive Process of the
   following form:

   x_{t+1} - ProcessMean =
     (x_t - ProcessMean) * ProcessScaling + ProcessSigma * w_t

   where w_t is zero-mean unit iid Gaussian noise. */
#define ProcessMean (-0.1)
#define ProcessScaling (0.4)
#define ProcessSigma (0.075)

/* The observation density is a mixture of Gaussians, where each
   observed object has a different sigma as follows. */
#define ObsSigma (0.03)

/* How many columns wide is the ASCII histogram? */
#define HistBins (79)
/* How many lines of text does the ASCII histogram take? */
#define HistLines (25)
/* What is the highest value the density histogram can represent
   before saturating above HistLines? */
#define MaxHistHeight (0.2)

/* What is the distance from the origin to the edge of the histogram? */
#define DisplayWidth (0.35)
