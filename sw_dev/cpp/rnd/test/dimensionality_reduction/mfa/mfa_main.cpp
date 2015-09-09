#include "mfa.hpp"
#include <gsl/gsl_blas.h>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

class my_vector_data_set_t : public vector_data_set_t
{
public:
    my_vector_data_set_t()  {}
    ~my_vector_data_set_t()  {}

    /*virtual*/ std::size_t length() const;
    /*virtual*/ void reset();
    /*virtual*/ bool get_next_vector(gsl_vector *vector_ptr);
};

/*virtual*/ std::size_t my_vector_data_set_t::length() const
{
    throw std::runtime_error("not yet implemented : my_vector_data_set_t::length()");
}

/*virtual*/ void my_vector_data_set_t::reset()
{
    throw std::runtime_error("not yet implemented : my_vector_data_set_t::reset()");
}

/*virtual*/ bool my_vector_data_set_t::get_next_vector(gsl_vector *vector_ptr)
{
    throw std::runtime_error("not yet implemented : my_vector_data_set_t::get_next_vector()");
}

}  // namespace local
}  // unnamed namespace

namespace my_mfa {

}  // namespace my_mfa

int mfa_main(int argc, char *argv[])
{
    {
        std::size_t num_mixture_components = 5;  // the number of mixture components, the arity of the latent discrete variable.
        std::size_t dim_latent_variable = 3;  // the dimensionality of the latent continuous variable.
        std::size_t dim_observed_variable = 20;  // the dimensionality of the observed continuous variable.
        const bool is_ppca = false;  // use MFA.
        mfa_t mfa(num_mixture_components, dim_latent_variable, dim_observed_variable, is_ppca);

        //
        const local::my_vector_data_set_t train_data;
        const double tol = 1e-1;
        const std::size_t max_iter = std::numeric_limits<std::size_t>::max();
        const bool result = mfa.em((vector_data_set_t &)train_data, tol, max_iter);

        //
        const gsl_vector *data1;
        const double llik = mfa.log_likelihood(data1);

        // Gets the expected data vector. (???)
        const gsl_vector *data2;
        const std::vector<bool> hidden_mask;
        gsl_vector *expected_data;
        mfa.get_expected_data(data2, hidden_mask, expected_data);

        //
        const std::string path;
        mfa.save(path);

        // Prints the W, mu, and sigma for the desired factor out to to disk in matlab ascii format.
        const std::size_t j = 0;
        const std::string file_name;
        mfa.print_W_to_file(j, file_name);
        mfa.print_mu_to_file(j, file_name);
        mfa.print_sigma_to_file(file_name);

        //
        mfa.convert_FA_to_PPCA();
    }

    {
        std::size_t num_mixture_components = 5;  // the number of mixture components, the arity of the latent discrete variable.
        std::size_t dim_latent_variable = 3;  // the dimensionality of the latent continuous variable.
        std::size_t dim_observed_variable = 20;  // the dimensionality of the observed continuous variable.
        const bool is_ppca = true;  // use PPCA.
        mfa_t ppca(num_mixture_components, dim_latent_variable, dim_observed_variable, is_ppca);

        const local::my_vector_data_set_t train_data;
        ppca.ppca_solve((vector_data_set_t &)train_data);
        //ppca.ppca_solve_fast((vector_data_set_t &)train_data);

        //
        const gsl_vector *data1;
        const double llik = ppca.log_likelihood(data1);

        //
        ppca.convert_PPCA_to_FA();
    }

	return 0;
}
