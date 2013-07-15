/* Some utility routines in utility.c */

double uniform_random(void);
double gaussian_random(void);
double evaluate_gaussian(double val, double sigma);

/* The model-specific functions from model_specific.c */

void initialise_model_specific_defaults(void);
void set_up_prior_conditions(void);
void obtain_observations(void);
void predict_sample_position(int new_sample, int old_sample);
double evaluate_observation_density(int new_sample);
void display_data(int iteration);
