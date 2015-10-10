#include "../mfa_lib/mfa.hpp"
#include <gsl/gsl_blas.h>
#include <boost/tokenizer.hpp>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <ctime>
#include <cassert>


namespace {
namespace local {

class my_vector_data_set_t : public vector_data_set_t
{
public:
    my_vector_data_set_t(const std::size_t dim_data, const std::size_t num_data);
    ~my_vector_data_set_t();

    /*virtual*/ std::size_t length() const;
    /*virtual*/ void reset();
    /*virtual*/ bool get_next_vector(gsl_vector *vector_ptr);

    bool load_data(const std::string &file_path, const bool is_colwise);

private:
    gsl_matrix *data_;  // column-major matrix.
    std::size_t index_;
};

my_vector_data_set_t::my_vector_data_set_t(const std::size_t dim_data, const std::size_t num_data)
: data_(NULL), index_(0)
{
	data_ = gsl_matrix_alloc(dim_data, num_data);
}

my_vector_data_set_t::~my_vector_data_set_t()
{
	gsl_matrix_free(data_);
	data_ = NULL;
}

/*virtual*/ std::size_t my_vector_data_set_t::length() const
{
    return data_ ? data_->size2 : 0;
}

/*virtual*/ void my_vector_data_set_t::reset()
{
    index_ = 0;
}

/*virtual*/ bool my_vector_data_set_t::get_next_vector(gsl_vector *vector_ptr)
{
    if (NULL == data_ || index_ >= length()) return false;

    gsl_matrix_get_col(vector_ptr, data_, index_);
    ++index_;
    return true;
}

bool my_vector_data_set_t::load_data(const std::string &file_path, const bool is_column_major)
{
#if defined(__GNUC__)
    std::ifstream stream(file_path.c_str());
#else
    std::ifstream stream(file_path);
#endif
	if (!stream)
	{
		std::cerr << "file not found : " << file_path << std::endl;
		return false;
	}

    const boost::char_separator<char> delimiter(", \t");

    gsl_vector *vec = gsl_vector_alloc(is_column_major ? data_->size2 : data_->size1);
    std::size_t line_idx = 0;
	std::string line;
	while (std::getline(stream, line))
	{
        if (!line.empty())
        {
            boost::tokenizer<boost::char_separator<char> > tokens(line, delimiter);

            std::size_t i = 0;
            for (boost::tokenizer<boost::char_separator<char> >::const_iterator cit = tokens.begin(); cit != tokens.end(); ++cit, ++i)
                gsl_vector_set(vec, i, atof(cit->c_str()));

            if (i != (is_column_major ? data_->size2 : data_->size1))
            {
                std::cerr << "unmatched number of sample" << std::endl;
                return false;
            }

            if (is_column_major) gsl_matrix_set_row(data_, line_idx, vec);
            else gsl_matrix_set_col(data_, line_idx, vec);
        }
        ++line_idx;
    }
    gsl_vector_free(vec);

    if (line_idx != (is_column_major ? data_->size1 : data_->size2))
    {
        std::cerr << "unmatched dimension of sample" << std::endl;
        return false;
    }

    return true;
}

}  // namespace local
}  // unnamed namespace

namespace my_mfa {

}  // namespace my_mfa

int mfa_main(int argc, char *argv[])
{
#if 0
    const std::string train_data_file("./data/machine_learning/swissroll_X_1.dat");  // #data = 20000.
    const bool is_column_major = true;  // a feature is a column vector.
    const std::size_t num_observed_data = 20000;  // the number of the observed data.
    const std::size_t dim_observed_variable = 3;  // the dimensionality of the observed continuous variable.
    const std::size_t dim_latent_variable = 2;  // the dimensionality of the latent continuous variable.
    const std::size_t num_mixture_components = 12;  // the number of mixture components, the arity of the latent discrete variable.
#elif 1
    const std::string train_data_file("./data/machine_learning/swissroll_X_2.dat");  // #data = 2000.
    const bool is_column_major = true;  // a feature is a column vector.
    const std::size_t num_observed_data = 2000;  // the number of the observed data.
    const std::size_t dim_observed_variable = 3;  // the dimensionality of the observed continuous variable.
    const std::size_t dim_latent_variable = 2;  // the dimensionality of the latent continuous variable.
    const std::size_t num_mixture_components = 1; //12;  // the number of mixture components, the arity of the latent discrete variable.
#elif 0
    const std::string train_data_file("./data/machine_learning/scurve_X_2.dat");  // #data = 2000.
    const bool is_column_major = true;  // a feature is a column vector.
    const std::size_t num_observed_data = 2000;  // the number of the observed data.
    const std::size_t dim_observed_variable = 3;  // the dimensionality of the observed continuous variable.
    const std::size_t dim_latent_variable = 2;  // the dimensionality of the latent continuous variable.
    const std::size_t num_mixture_components = 12;  // the number of mixture components, the arity of the latent discrete variable.
#elif 0
    const std::string train_data_file("./data/machine_learning/scurve_X_3.dat");  // #data = 2000.
    const bool is_column_major = false;  // a feature is a row vector.
    const std::size_t num_observed_data = 2000;  // the number of the observed data.
    const std::size_t dim_observed_variable = 3;  // the dimensionality of the observed continuous variable.
    const std::size_t dim_latent_variable = 2;  // the dimensionality of the latent continuous variable.
    const std::size_t num_mixture_components = 12;  // the number of mixture components, the arity of the latent discrete variable.
#endif

    {
        const bool is_ppca = false;  // use MFA.

        // load training data.
        local::my_vector_data_set_t train_data(dim_observed_variable, num_observed_data);
        if (!train_data.load_data(train_data_file, is_column_major))
        {
            std::cerr << "file load error : " << train_data_file << std::endl;
            return false;
        }

        // create MFA.
        mfa_t mfa(num_mixture_components, dim_latent_variable, dim_observed_variable, is_ppca);

        // train MFA.
        {
            // for elapsed time.
            const std::time_t start_time = std::time(NULL);

            const double tol = 1e-1;
            const std::size_t max_iter = std::numeric_limits<std::size_t>::max();
            const bool result = mfa.em((vector_data_set_t &)train_data, tol, max_iter);

            std::cout << "the result of EM = " << result << std::endl;

            // for elapsed time.
            const std::time_t finish_time = std::time(NULL);
            const double elapsed_time = std::difftime(finish_time, start_time);
            std::cout << "elapsed time : " << elapsed_time << " seconds" << std::endl;
        }

        // compute likelihood.
        {
            gsl_vector *data = gsl_vector_alloc(dim_observed_variable);
            // the first datum in ./data/machine_learning/swissroll_X_2.dat.
            gsl_vector_set(data, 0, 3.6981235e+00);
            gsl_vector_set(data, 1, 5.4035790e+00);
            gsl_vector_set(data, 2, 1.3365016e+01);

            const double llik = mfa.log_likelihood(data);
            std::cout << "log likelihood = " << llik << std::endl;

            gsl_vector_free(data);
            data = NULL;
        }

        // Gets the expected data vector ==> reconstructed data. (???)
#if 0
        {
            gsl_vector *data = gsl_vector_alloc(dim_observed_variable);
            // the first datum in ./data/machine_learning/swissroll_X_2.dat.
            gsl_vector_set(data, 0, 3.6981235e+00);
            gsl_vector_set(data, 1, 5.4035790e+00);
            gsl_vector_set(data, 2, 1.3365016e+01);

            const std::vector<bool> hidden_mask(dim_observed_variable, true);  // ???
            gsl_vector *expected_data = NULL;//gsl_vector_alloc(dim_observed_variable);

            // NOTE [caution] >> unknown error occurred.
            mfa.get_expected_data(data, hidden_mask, expected_data);

            gsl_vector_free(data);
            data = NULL;
            gsl_vector_free(expected_data);
            expected_data = NULL;
        }
#endif

        // save MFA model.
        const std::string model_path("./data/machine_learning/mfa/mfa_test.model");
        mfa.save(model_path);

/*
        // Prints the W, mu, and sigma for the desired factor out to to disk in matlab ascii format.
        const std::size_t j = 0;
        const std::string W_file_path(""), mu_file_path(""), sigma_file_path("");
        mfa.print_W_to_file(j, W_file_path);
        mfa.print_mu_to_file(j, mu_file_path);
        mfa.print_sigma_to_file(sigma_file_path);

        // convert MFA to MPPCA.
        mfa.convert_FA_to_PPCA();
*/
    }

    {
        const bool is_ppca = true;  // use MPPCA.

        // load training data.
        local::my_vector_data_set_t train_data(dim_observed_variable, num_observed_data);
        if (!train_data.load_data(train_data_file, is_column_major))
        {
            std::cerr << "file load error : " << train_data_file << std::endl;
            return false;
        }

        // create MPPCA.
        mfa_t mppca(num_mixture_components, dim_latent_variable, dim_observed_variable, is_ppca);

        // train MPPCA.
        {
            // for elapsed time.
            const std::time_t start_time = std::time(NULL);

            // NOTE [caution] >> limitation of this implementation of PPCA.
            //  PPCA only works for one mixture component.
            assert(1 == num_mixture_components);

            mppca.ppca_solve((vector_data_set_t &)train_data);
            //mppca.ppca_solve_fast((vector_data_set_t &)train_data);

            // for elapsed time.
            const std::time_t finish_time = std::time(NULL);
            const double elapsed_time = std::difftime(finish_time, start_time);
            std::cout << "elapsed time : " << elapsed_time << " seconds" << std::endl;
        }

        // compute likelihood.
        {
            gsl_vector *data = gsl_vector_alloc(dim_observed_variable);
            // the first datum in ./data/machine_learning/swissroll_X_2.dat.
            gsl_vector_set(data, 0, 3.6981235e+00);
            gsl_vector_set(data, 1, 5.4035790e+00);
            gsl_vector_set(data, 2, 1.3365016e+01);

            const double llik = mppca.log_likelihood(data);
            std::cout << "log likelihood = " << llik << std::endl;

            gsl_vector_free(data);
            data = NULL;
        }

        // convert MPPCA to MFA.
        //mppca.convert_PPCA_to_FA();
    }

	return 0;
}
