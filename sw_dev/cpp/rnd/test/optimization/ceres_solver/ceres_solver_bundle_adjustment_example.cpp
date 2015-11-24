//#include "stdafx.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <iostream>
#include <string>
#include <cmath>
#include <cstdio>


namespace {
namespace local {

// Read a Bundle Adjustment in the Large dataset.
class BALProblem
{
public:
    ~BALProblem()
    {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

public:
    int num_observations() const  { return num_observations_; }
    const double * observations() const  { return observations_; }
    double * mutable_cameras()  { return parameters_; }
    double * mutable_points()  { return parameters_  + 9 * num_cameras_; }
    double * mutable_camera_for_observation(int i)
    {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double * mutable_point_for_observation(int i)
    {
        return mutable_points() + point_index_[i] * 3;
    }

    bool LoadFile(const char *filename)
    {
        FILE *fptr = fopen(filename, "r");
        if (NULL == fptr)
        {
            return false;
        };

        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);

        point_index_ = new int [num_observations_];
        camera_index_ = new int [num_observations_];
        observations_ = new double [2 * num_observations_];
        num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
        parameters_ = new double [num_parameters_];

        for (int i = 0; i < num_observations_; ++i)
        {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j)
            {
                FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
            }
        }
        for (int i = 0; i < num_parameters_; ++i)
        {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }

        return true;
    }

private:
    template<typename T>
    void FscanfOrDie(FILE *fptr, const char *format, T *value)
    {
        int num_scanned = fscanf(fptr, format, value);
        if (1 != num_scanned)
        {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }

private:
    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    int *point_index_;
    int *camera_index_;
    double *observations_;
    double *parameters_;
};

struct SnavelyReprojectionError
{
public:
    SnavelyReprojectionError(double observed_x, double observed_y)
    : observed_x(observed_x), observed_y(observed_y)
    {}

public:
    template <typename T>
    bool operator()(const T * const camera, const T * const point, T *residuals) const
    {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion.
        const T &l1 = camera[7];
        const T &l2 = camera[8];
        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + r2  * (l1 + l2  * r2);

        // Compute final projected point position.
        const T &focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction * Create(const double observed_x, const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(new SnavelyReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};

}  // namespace local
}  // unnamed namespace

namespace my_ceres_solver {

// REF [site] >> https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/simple_bundle_adjuster.cc
void bundle_adjustment_example()
{
    // BAL dataset
    // REF [site] >> http://grail.cs.washington.edu/projects/bal/
    const std::string filename("./data/machine_vision/bundle_adjustment/problem-49-7776-pre.txt");

    local::BALProblem bal_problem;
    if (!bal_problem.LoadFile(filename.c_str()))
    {
        std::cerr << "ERROR: unable to open file " << filename << std::endl;
        return;
    }

    const double *observations = bal_problem.observations();

    // Create residuals for each observation in the bundle adjustment problem.
    // The parameters for cameras and points are added automatically.
    ceres::Problem problem;
    for (int i = 0; i < bal_problem.num_observations(); ++i)
    {
        // Each Residual block takes a point and a camera as input and outputs a 2 dimensional residual.
        // Internally, the cost function stores the observed image location and compares the reprojection against the observation.
        ceres::CostFunction *cost_function = local::SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);
        problem.AddResidualBlock(
            cost_function,
            NULL /* squared loss */,
            bal_problem.mutable_camera_for_observation(i),
            bal_problem.mutable_point_for_observation(i)
        );
    }

    // Make Ceres automatically detect the bundle structure.
    // Note that the standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;
}

}  // namespace my_ceres_solver
