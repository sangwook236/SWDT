//#include "stdafx.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <thread>
#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "bal_problem.h"


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

	// File contents:
	//	#cameras = 49
	//	#points = 7776
	//	#observations = 31843
	//	observation = [ camera id, point id, x, y ]
	//	parameters : #parameters = 9 * #cameras + 3 * #points = 23769
	//		camera parameters = [ angle-axis rotation (3), translation (3), second and fourth order radial distortion (2), focal length (1) (?) ]
	bool LoadFile(const char *filename)
	{
		FILE *fptr = fopen(filename, "r");
		if (nullptr == fptr)
		{
			return false;
		}

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

		// Compute the center of distortion.
		// The sign change comes from the camera model that Noah Snavely's Bundler assumes, whereby the camera coordinate system has a negative z axis.
		T xp = -p[0] / p[2];
		T yp = -p[1] / p[2];

		// Apply second and fourth order radial distortion.
		const T &l1 = camera[7];
		const T &l2 = camera[8];
		T r2 = xp * xp + yp * yp;
		T distortion = T(1.0) + r2 * (l1 + l2 * r2);

		// Compute final projected point position.
		const T &focal = camera[6];
		T predicted_x = focal * distortion * xp;
		T predicted_y = focal * distortion * yp;

		// The error is the difference between the predicted and observed position.
		residuals[0] = predicted_x - T(observed_x);
		residuals[1] = predicted_y - T(observed_y);
		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction * Create(const double observed_x, const double observed_y)
	{
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(new SnavelyReprojectionError(observed_x, observed_y)));
	}

private:
	double observed_x;
	double observed_y;
};

struct SnavelyReprojectionErrorWithQuaternions
{
	// (u, v): the position of the observation with respect to the image center point.
	SnavelyReprojectionErrorWithQuaternions(double observed_x, double observed_y)
	: observed_x(observed_x), observed_y(observed_y)
	{}

	template <typename T>
	bool operator()(const T* const camera, const T* const point, T* residuals) const
	{
		// camera[0,1,2,3] is are the rotation of the camera as a quaternion.
		//
		// We use QuaternionRotatePoint as it does not assume that the quaternion is normalized,
		// since one of the ways to run the bundle adjuster is to let Ceres optimize all 4 quaternion parameters without using a Quaternion manifold.
		T p[3];
		ceres::QuaternionRotatePoint(camera, point, p);

		p[0] += camera[4];
		p[1] += camera[5];
		p[2] += camera[6];

		// Compute the center of distortion.
		// The sign change comes from the camera model that Noah Snavely's Bundler assumes, whereby the camera coordinate system has a negative z axis.
		const T xp = -p[0] / p[2];
		const T yp = -p[1] / p[2];

		// Apply second and fourth order radial distortion.
		const T& l1 = camera[8];
		const T& l2 = camera[9];

		const T r2 = xp * xp + yp * yp;
		const T distortion = 1.0 + r2 * (l1 + l2 * r2);

		// Compute final projected point position.
		const T& focal = camera[7];
		const T predicted_x = focal * distortion * xp;
		const T predicted_y = focal * distortion * yp;

		// The error is the difference between the predicted and observed position.
		residuals[0] = predicted_x - observed_x;
		residuals[1] = predicted_y - observed_y;

		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(const double observed_x, const double observed_y)
	{
		return (
			new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorWithQuaternions, 2, 10, 3>(
				new SnavelyReprojectionErrorWithQuaternions(observed_x, observed_y)
			)
		);
	}

	double observed_x;
	double observed_y;
};

DEFINE_string(input, "../problem-49-7776-pre.txt", "Input File name");
DEFINE_string(trust_region_strategy, "levenberg_marquardt", "Options are: levenberg_marquardt, dogleg.");
DEFINE_string(dogleg, "traditional_dogleg", "Options are: traditional_dogleg, subspace_dogleg.");
DEFINE_bool(inner_iterations, false, "Use inner iterations to non-linearly refine each successful trust region step.");
DEFINE_string(blocks_for_inner_iterations, "automatic", "Options are: automatic, cameras, points, cameras,points, points,cameras");
DEFINE_string(linear_solver, "sparse_schur", "Options are: sparse_schur, dense_schur, iterative_schur, sparse_normal_cholesky, dense_qr, dense_normal_cholesky, and cgnr.");
DEFINE_bool(explicit_schur_complement, false, "If using ITERATIVE_SCHUR then explicitly compute the Schur complement.");
DEFINE_string(preconditioner, "jacobi", "Options are: identity, jacobi, schur_jacobi, schur_power_series_expansion, cluster_jacobi, cluster_tridiagonal.");
DEFINE_string(visibility_clustering, "canonical_views", "single_linkage, canonical_views");
DEFINE_bool(use_spse_initialization, false, "Use power series expansion to initialize the solution in ITERATIVE_SCHUR linear solver.");
DEFINE_string(sparse_linear_algebra_library, "suite_sparse", "Options are: suite_sparse, accelerate_sparse, eigen_sparse, and cuda_sparse.");
DEFINE_string(dense_linear_algebra_library, "eigen", "Options are: eigen, lapack, and cuda");
DEFINE_string(ordering_type, "amd", "Options are: amd, nesdis");
DEFINE_string(linear_solver_ordering, "user", "Options are: automatic and user");
DEFINE_bool(use_quaternions, false, "If true, uses quaternions to represent rotations. If false, angle axis is used.");
DEFINE_bool(use_manifolds, false, "For quaternions, use a manifold.");
DEFINE_bool(robustify, false, "Use a robust loss function.");
DEFINE_double(eta, 1e-2, "Default value for eta. Eta determines the accuracy of each linear solve of the truncated newton step. Changing this parameter can affect solve performance.");
DEFINE_int32(num_threads, -1, "Number of threads. -1 = std::thread::hardware_concurrency.");
DEFINE_int32(num_iterations, 5, "Number of iterations.");
DEFINE_int32(max_linear_solver_iterations, 500, "Maximum number of iterations  for solution of linear system.");
DEFINE_double(spse_tolerance, 0.1, "Tolerance to reach during the iterations of power series expansion initialization or preconditioning.");
DEFINE_int32(max_num_spse_iterations, 5, "Maximum number of iterations for power series expansion initialization or preconditioning.");
DEFINE_double(max_solver_time, 1e32, "Maximum solve time in seconds.");
DEFINE_bool(nonmonotonic_steps, false, "Trust region algorithm can use nonmonotic steps.");
DEFINE_double(rotation_sigma, 0.0, "Standard deviation of camera rotation perturbation.");
DEFINE_double(translation_sigma, 0.0, "Standard deviation of the camera translation perturbation.");
DEFINE_double(point_sigma, 0.0, "Standard deviation of the point perturbation.");
DEFINE_int32(random_seed, 38401, "Random seed used to set the state of the pseudo random number generator used to generate the perturbations.");
DEFINE_bool(line_search, false, "Use a line search instead of trust region algorithm.");
DEFINE_bool(mixed_precision_solves, false, "Use mixed precision solves.");
DEFINE_int32(max_num_refinement_iterations, 0, "Iterative refinement iterations");
DEFINE_string(initial_ply, "", "Export the BAL file data as a PLY file.");
DEFINE_string(final_ply, "", "Export the refined BAL file data as a PLY file.");

void SetLinearSolver(ceres::Solver::Options* options)
{
	CHECK(StringToLinearSolverType(CERES_GET_FLAG(FLAGS_linear_solver), &options->linear_solver_type));
	CHECK(StringToPreconditionerType(CERES_GET_FLAG(FLAGS_preconditioner), &options->preconditioner_type));
	CHECK(StringToVisibilityClusteringType(CERES_GET_FLAG(FLAGS_visibility_clustering), &options->visibility_clustering_type));
	CHECK(StringToSparseLinearAlgebraLibraryType(CERES_GET_FLAG(FLAGS_sparse_linear_algebra_library), &options->sparse_linear_algebra_library_type));
	CHECK(StringToDenseLinearAlgebraLibraryType(CERES_GET_FLAG(FLAGS_dense_linear_algebra_library), &options->dense_linear_algebra_library_type));
	CHECK(StringToLinearSolverOrderingType(CERES_GET_FLAG(FLAGS_ordering_type), &options->linear_solver_ordering_type));
	options->use_explicit_schur_complement = CERES_GET_FLAG(FLAGS_explicit_schur_complement);
	options->use_mixed_precision_solves = CERES_GET_FLAG(FLAGS_mixed_precision_solves);
	options->max_num_refinement_iterations = CERES_GET_FLAG(FLAGS_max_num_refinement_iterations);
	options->max_linear_solver_iterations = CERES_GET_FLAG(FLAGS_max_linear_solver_iterations);
	options->use_spse_initialization = CERES_GET_FLAG(FLAGS_use_spse_initialization);
	options->spse_tolerance = CERES_GET_FLAG(FLAGS_spse_tolerance);
	options->max_num_spse_iterations = CERES_GET_FLAG(FLAGS_max_num_spse_iterations);
}

void SetOrdering(ceres::examples::BALProblem* bal_problem, ceres::Solver::Options* options)
{
	const int num_points = bal_problem->num_points();
	const int point_block_size = bal_problem->point_block_size();
	double* points = bal_problem->mutable_points();
	const int num_cameras = bal_problem->num_cameras();
	const int camera_block_size = bal_problem->camera_block_size();
	double* cameras = bal_problem->mutable_cameras();
	if (options->use_inner_iterations)
	{
		if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) == "cameras")
		{
			LOG(INFO) << "Camera blocks for inner iterations";
			options->inner_iteration_ordering = std::make_shared<ceres::ParameterBlockOrdering>();
			for (int i = 0; i < num_cameras; ++i)
			{
				options->inner_iteration_ordering->AddElementToGroup(cameras + camera_block_size * i, 0);
			}
		}
		else if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) == "points")
		{
			LOG(INFO) << "Point blocks for inner iterations";
			options->inner_iteration_ordering = std::make_shared<ceres::ParameterBlockOrdering>();
			for (int i = 0; i < num_points; ++i)
			{
				options->inner_iteration_ordering->AddElementToGroup(points + point_block_size * i, 0);
			}
		}
		else if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) == "cameras,points")
		{
			LOG(INFO) << "Camera followed by point blocks for inner iterations";
			options->inner_iteration_ordering = std::make_shared<ceres::ParameterBlockOrdering>();
			for (int i = 0; i < num_cameras; ++i)
			{
				options->inner_iteration_ordering->AddElementToGroup(cameras + camera_block_size * i, 0);
			}
			for (int i = 0; i < num_points; ++i)
			{
				options->inner_iteration_ordering->AddElementToGroup(points + point_block_size * i, 1);
			}
		}
		else if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) == "points,cameras")
		{
			LOG(INFO) << "Point followed by camera blocks for inner iterations";
			options->inner_iteration_ordering = std::make_shared<ceres::ParameterBlockOrdering>();
			for (int i = 0; i < num_cameras; ++i)
			{
				options->inner_iteration_ordering->AddElementToGroup(cameras + camera_block_size * i, 1);
			}
			for (int i = 0; i < num_points; ++i)
			{
				options->inner_iteration_ordering->AddElementToGroup(points + point_block_size * i, 0);
			}
		}
		else if (CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations) == "automatic")
		{
			LOG(INFO) << "Choosing automatic blocks for inner iterations";
		}
		else
		{
			LOG(FATAL) << "Unknown block type for inner iterations: " << CERES_GET_FLAG(FLAGS_blocks_for_inner_iterations);
		}
	}

	// Bundle adjustment problems have a sparsity structure that makes them amenable to more specialized and much more efficient solution strategies.
	// The SPARSE_SCHUR, DENSE_SCHUR and ITERATIVE_SCHUR solvers make use of this specialized structure.
	//
	// This can either be done by specifying a Options::linear_solver_ordering or having Ceres figure it out automatically using a greedy maximum independent set algorithm.
	if (CERES_GET_FLAG(FLAGS_linear_solver_ordering) == "user")
	{
		auto* ordering = new ceres::ParameterBlockOrdering;
		// The points come before the cameras.
		for (int i = 0; i < num_points; ++i)
		{
			ordering->AddElementToGroup(points + point_block_size * i, 0);
		}
		for (int i = 0; i < num_cameras; ++i)
		{
			// When using axis-angle, there is a single parameter block for the entire camera.
			ordering->AddElementToGroup(cameras + camera_block_size * i, 1);
		}
		options->linear_solver_ordering.reset(ordering);
	}
}

void SetMinimizerOptions(ceres::Solver::Options* options)
{
	options->max_num_iterations = CERES_GET_FLAG(FLAGS_num_iterations);
	options->minimizer_progress_to_stdout = true;
	if (CERES_GET_FLAG(FLAGS_num_threads) == -1)
	{
		const int num_available_threads = static_cast<int>(std::thread::hardware_concurrency());
		if (num_available_threads > 0)
		{
			options->num_threads = num_available_threads;
		}
	}
	else
	{
		options->num_threads = CERES_GET_FLAG(FLAGS_num_threads);
	}
	CHECK_GE(options->num_threads, 1);

	options->eta = CERES_GET_FLAG(FLAGS_eta);
	options->max_solver_time_in_seconds = CERES_GET_FLAG(FLAGS_max_solver_time);
	options->use_nonmonotonic_steps = CERES_GET_FLAG(FLAGS_nonmonotonic_steps);
	if (CERES_GET_FLAG(FLAGS_line_search))
	{
		options->minimizer_type = ceres::LINE_SEARCH;
	}

	CHECK(StringToTrustRegionStrategyType(CERES_GET_FLAG(FLAGS_trust_region_strategy), &options->trust_region_strategy_type));
	CHECK(StringToDoglegType(CERES_GET_FLAG(FLAGS_dogleg), &options->dogleg_type));
	options->use_inner_iterations = CERES_GET_FLAG(FLAGS_inner_iterations);
}

void SetSolverOptionsFromFlags(ceres::examples::BALProblem* bal_problem, ceres::Solver::Options* options)
{
	SetMinimizerOptions(options);
	SetLinearSolver(options);
	SetOrdering(bal_problem, options);
}

void BuildProblem(ceres::examples::BALProblem* bal_problem, ceres::Problem* problem)
{
	const int point_block_size = bal_problem->point_block_size();
	const int camera_block_size = bal_problem->camera_block_size();
	double* points = bal_problem->mutable_points();
	double* cameras = bal_problem->mutable_cameras();

	// Observations is 2*num_observations long array observations = [u_1, u_2, ... , u_n],
	// where each u_i is two dimensional, the x and y positions of the observation.
	const double* observations = bal_problem->observations();
	for (int i = 0; i < bal_problem->num_observations(); ++i)
	{
		// Each Residual block takes a point and a camera as input and outputs a 2 dimensional residual.
		ceres::CostFunction* cost_function = (CERES_GET_FLAG(FLAGS_use_quaternions))
			? SnavelyReprojectionErrorWithQuaternions::Create(observations[2 * i + 0], observations[2 * i + 1])
			: SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);

		// If enabled use Huber's loss function.
		ceres::LossFunction* loss_function = CERES_GET_FLAG(FLAGS_robustify) ? new ceres::HuberLoss(1.0) : nullptr;

		// Each observation corresponds to a pair of a camera and a point
		// which are identified by camera_index()[i] and point_index()[i] respectively.
		double* camera = cameras + camera_block_size * bal_problem->camera_index()[i];
		double* point = points + point_block_size * bal_problem->point_index()[i];
		problem->AddResidualBlock(cost_function, loss_function, camera, point);
	}

	if (CERES_GET_FLAG(FLAGS_use_quaternions) && CERES_GET_FLAG(FLAGS_use_manifolds))
	{
		ceres::Manifold* camera_manifold = new ceres::ProductManifold<ceres::QuaternionManifold, ceres::EuclideanManifold<6>>{};
		for (int i = 0; i < bal_problem->num_cameras(); ++i)
		{
			problem->SetManifold(cameras + camera_block_size * i, camera_manifold);
		}
	}
}

void SolveProblem(const char* filename)
{
	srand(CERES_GET_FLAG(FLAGS_random_seed));

	ceres::examples::BALProblem bal_problem(filename, CERES_GET_FLAG(FLAGS_use_quaternions));

	if (!CERES_GET_FLAG(FLAGS_initial_ply).empty())
	{
		bal_problem.WriteToPLYFile(CERES_GET_FLAG(FLAGS_initial_ply));
	}

	bal_problem.Normalize();
	bal_problem.Perturb(CERES_GET_FLAG(FLAGS_rotation_sigma), CERES_GET_FLAG(FLAGS_translation_sigma), CERES_GET_FLAG(FLAGS_point_sigma));

	ceres::Problem problem;
	BuildProblem(&bal_problem, &problem);

	ceres::Solver::Options options;
	SetSolverOptionsFromFlags(&bal_problem, &options);
	options.gradient_tolerance = 1e-16;
	options.function_tolerance = 1e-16;
	options.parameter_tolerance = 1e-16;

	ceres::Solver::Summary summary;
	Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	if (!CERES_GET_FLAG(FLAGS_final_ply).empty())
	{
		bal_problem.WriteToPLYFile(CERES_GET_FLAG(FLAGS_final_ply));
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_ceres_solver {

// REF [site] >> https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/simple_bundle_adjuster.cc
void simple_bundle_adjustment_example()
{
	// BAL dataset.
	// REF [paper] >> "Bundle Adjustment in the Large", ECCV 2010.
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
			nullptr,  // Squared loss.
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

// REF [site] >>
//	https://github.com/ceres-solver/ceres-solver/blob/master/examples/bundle_adjuster.cc
//	https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/bundle_adjuster.cc
void bundle_adjustment_example()
{
	if (CERES_GET_FLAG(local::FLAGS_input).empty())
	{
		LOG(ERROR) << "Usage: bundle_adjuster --input=bal_problem";
		return;
	}

	CHECK(CERES_GET_FLAG(local::FLAGS_use_quaternions) || !CERES_GET_FLAG(local::FLAGS_use_manifolds))
		<< "--use_manifolds can only be used with --use_quaternions.";
	local::SolveProblem(CERES_GET_FLAG(local::FLAGS_input).c_str());
}

}  // namespace my_ceres_solver
