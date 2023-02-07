//#include "stdafx.h"
#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>


namespace {
namespace local {

DEFINE_double(corridor_length, 30.0, "Length of the corridor that the robot is travelling down.");
DEFINE_double(pose_separation, 0.5, "The distance that the robot traverses between successive odometry updates.");
DEFINE_double(odometry_stddev, 0.1, "The standard deviation of odometry error of the robot.");
DEFINE_double(range_stddev, 0.01, "The standard deviation of range readings of the robot.");

// The stride length of the dynamic_autodiff_cost_function evaluator.
static constexpr int kStride = 10;

struct OdometryConstraint
{
	using OdometryCostFunction = ceres::AutoDiffCostFunction<OdometryConstraint, 1, 1>;

	OdometryConstraint(double odometry_mean, double odometry_stddev)
	: odometry_mean(odometry_mean), odometry_stddev(odometry_stddev)
	{}

	template <typename T>
	bool operator()(const T* const odometry, T* residual) const
	{
		*residual = (*odometry - odometry_mean) / odometry_stddev;
		return true;
	}

	static OdometryCostFunction* Create(const double odometry_value)
	{
		return new OdometryCostFunction(new OdometryConstraint(odometry_value, CERES_GET_FLAG(FLAGS_odometry_stddev)));
	}

	const double odometry_mean;
	const double odometry_stddev;
};

struct RangeConstraint
{
	using RangeCostFunction = ceres::DynamicAutoDiffCostFunction<RangeConstraint, kStride>;

	RangeConstraint(int pose_index, double range_reading, double range_stddev, double corridor_length)
	: pose_index(pose_index), range_reading(range_reading), range_stddev(range_stddev), corridor_length(corridor_length)
	{}

	template <typename T>
	bool operator()(T const* const* relative_poses, T* residuals) const
	{
		T global_pose(0);
		for (int i = 0; i <= pose_index; ++i)
		{
			global_pose += relative_poses[i][0];
		}
		residuals[0] = (global_pose + range_reading - corridor_length) / range_stddev;
		return true;
	}

	// Factory method to create a CostFunction from a RangeConstraint to conveniently add to a ceres problem.
	static RangeCostFunction* Create(const int pose_index, const double range_reading, std::vector<double>* odometry_values, std::vector<double*>* parameter_blocks)
	{
		auto* constraint = new RangeConstraint(pose_index, range_reading, CERES_GET_FLAG(FLAGS_range_stddev), CERES_GET_FLAG(FLAGS_corridor_length));
		auto* cost_function = new RangeCostFunction(constraint);
		// Add all the parameter blocks that affect this constraint.
		parameter_blocks->clear();
		for (int i = 0; i <= pose_index; ++i)
		{
			parameter_blocks->push_back(&((*odometry_values)[i]));
			cost_function->AddParameterBlock(1);
		}
		cost_function->SetNumResiduals(1);
		return cost_function;
	}

	const int pose_index;
	const double range_reading;
	const double range_stddev;
	const double corridor_length;
};

void SimulateRobot(std::vector<double>* odometry_values, std::vector<double>* range_readings)
{
	const int num_steps = static_cast<int>(ceil(CERES_GET_FLAG(FLAGS_corridor_length) / CERES_GET_FLAG(FLAGS_pose_separation)));
	std::mt19937 prng;
	std::normal_distribution<double> odometry_noise(0.0, CERES_GET_FLAG(FLAGS_odometry_stddev));
	std::normal_distribution<double> range_noise(0.0, CERES_GET_FLAG(FLAGS_range_stddev));

	// The robot starts out at the origin.
	double robot_location = 0.0;
	for (int i = 0; i < num_steps; ++i)
	{
		const double actual_odometry_value = std::min(CERES_GET_FLAG(FLAGS_pose_separation), CERES_GET_FLAG(FLAGS_corridor_length) - robot_location);
		robot_location += actual_odometry_value;
		const double actual_range = CERES_GET_FLAG(FLAGS_corridor_length) - robot_location;
		const double observed_odometry = actual_odometry_value + odometry_noise(prng);
		const double observed_range = actual_range + range_noise(prng);
		odometry_values->push_back(observed_odometry);
		range_readings->push_back(observed_range);
	}
}

void PrintState(const std::vector<double>& odometry_readings, const std::vector<double>& range_readings)
{
	CHECK_EQ(odometry_readings.size(), range_readings.size());
	double robot_location = 0.0;
	printf("pose: location     odom    range  r.error  o.error\n");
	for (size_t i = 0; i < odometry_readings.size(); ++i)
	{
		robot_location += odometry_readings[i];
		const double range_error = robot_location + range_readings[i] - CERES_GET_FLAG(FLAGS_corridor_length);
		const double odometry_error = CERES_GET_FLAG(FLAGS_pose_separation) - odometry_readings[i];
		printf("%4d: %8.3f %8.3f %8.3f %8.3f %8.3f\n",
			static_cast<int>(i),
			robot_location,
			odometry_readings[i],
			range_readings[i],
			range_error,
			odometry_error
		);
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_ceres_solver {

// REF [site] >> https://github.com/ceres-solver/ceres-solver/blob/master/examples/robot_pose_mle.cc
void robot_pose_mle_example()
{
	// Make sure that the arguments parsed are all positive.
	CHECK_GT(CERES_GET_FLAG(local::FLAGS_corridor_length), 0.0);
	CHECK_GT(CERES_GET_FLAG(local::FLAGS_pose_separation), 0.0);
	CHECK_GT(CERES_GET_FLAG(local::FLAGS_odometry_stddev), 0.0);
	CHECK_GT(CERES_GET_FLAG(local::FLAGS_range_stddev), 0.0);

	std::vector<double> odometry_values;
	std::vector<double> range_readings;
	local::SimulateRobot(&odometry_values, &range_readings);

	std::cout << "Initial values:" << std::endl;
	local::PrintState(odometry_values, range_readings);

	ceres::Problem problem;
	for (int i = 0; i < (int)odometry_values.size(); ++i)
	{
		// Create and add a DynamicAutoDiffCostFunction for the RangeConstraint from pose i.
		std::vector<double*> parameter_blocks;
		local::RangeConstraint::RangeCostFunction* range_cost_function =
			local::RangeConstraint::Create(i, range_readings[i], &odometry_values, &parameter_blocks);
		problem.AddResidualBlock(range_cost_function, nullptr, parameter_blocks);

		// Create and add an AutoDiffCostFunction for the OdometryConstraint for pose i.
		problem.AddResidualBlock(local::OdometryConstraint::Create(odometry_values[i]), nullptr, &(odometry_values[i]));
	}

	ceres::Solver::Options solver_options;
	solver_options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	std::cout << "Solving..." << std::endl;
	ceres::Solve(solver_options, &problem, &summary);
	std::cout << "Done." << std::endl;

	std::cout << summary.FullReport() << std::endl;

	std::cout << "Final values:" << std::endl;
	local::PrintState(odometry_values, range_readings);
}

}  // namespace my_ceres_solver
