//#include "stdafx.h"
#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#ifdef G2O_USE_VENDORED_CERES
#include <g2o/EXTERNAL/ceres/autodiff.h>
#else
#include <ceres/internal/autodiff.h>
#endif
#include <g2o/core/auto_differentiation.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/batch_stats.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#if defined G2O_HAVE_CHOLMOD
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#else
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#endif

namespace g2o {
namespace bal {

using Vector9 = VectorN<9>;

}  // namespace bal
}  // namespace g2o

namespace {
namespace local {

/**
 * \brief camera vertex which stores the parameters for a pinhole camera
 *
 * The parameters of the camera are
 * - rx,ry,rz representing the rotation axis, whereas the angle is given by
 * ||(rx,ry,rz)||
 * - tx,ty,tz the translation of the camera
 * - f the focal length of the camera
 * - k1, k2 two radial distortion parameters
 */
class VertexCameraBAL : public g2o::BaseVertex<9, g2o::bal::Vector9>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	VertexCameraBAL() {}

	virtual bool read(std::istream& /*is*/)
	{
		std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
		return false;
	}

	virtual bool write(std::ostream& /*os*/) const
	{
		std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
		return false;
	}

	virtual void setToOriginImpl()
	{
		std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
	}

	virtual void oplusImpl(const double* update)
	{
		const g2o::bal::Vector9::ConstMapType v(update, VertexCameraBAL::Dimension);
		_estimate += v;
	}
};

/**
 * \brief 3D world feature
 *
 * A 3D point feature in the world
 */
class VertexPointBAL : public g2o::BaseVertex<3, g2o::Vector3>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	VertexPointBAL() {}

	bool read(std::istream& /*is*/) override
	{
		std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
		return false;
	}

	bool write(std::ostream& /*os*/) const override
	{
		std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
		return false;
	}

	void setToOriginImpl() override
	{
		std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
	}

	void oplusImpl(const double* update) override
	{
		const g2o::Vector3::ConstMapType v(update);
		_estimate += v;
	}
};

/**
 * \brief edge representing the observation of a world feature by a camera
 *
 * see: http://grail.cs.washington.edu/projects/bal/
 * We use a pinhole camera model; the parameters we estimate for each camera
 * area rotation R, a translation t, a focal length f and two radial distortion
 * parameters k1 and k2. The formula for projecting a 3D point X into a camera
 * R,t,f,k1,k2 is:
 * P  =  R * X + t     (conversion from world to camera coordinates)
 * p  = -P / P.z       (perspective division)
 * p' =  f * r(p) * p  (conversion to pixel coordinates) where P.z is the third
 * (z) coordinate of P.
 *
 * In the last equation, r(p) is a function that computes a scaling factor to
 * undo the radial distortion: r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.
 *
 * This gives a projection in pixels, where the origin of the image is the
 * center of the image, the positive x-axis points right, and the positive
 * y-axis points up (in addition, in the camera coordinate system, the positive
 * z-axis points backwards, so the camera is looking down the negative z-axis,
 * as in OpenGL).
 */
class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, g2o::Vector2, VertexCameraBAL, VertexPointBAL>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	EdgeObservationBAL() {}

	bool read(std::istream& /*is*/) override
	{
		std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
		return false;
	}

	bool write(std::ostream& /*os*/) const override
	{
		std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
		return false;
	}

	/**
	 * templatized function to compute the error as described in the comment above
	 */
	template <typename T>
	bool operator()(const T* p_camera, const T* p_point, T* p_error) const
	{
		typename g2o::VectorN<9, T>::ConstMapType camera(p_camera);
		typename g2o::VectorN<3, T>::ConstMapType point(p_point);

		typename g2o::VectorN<3, T> p;

		// Rodrigues' formula for the rotation.
		T theta = camera.template head<3>().norm();
		if (theta > T(0))
		{
			g2o::VectorN<3, T> v = camera.template head<3>() / theta;
			T cth = cos(theta);
			T sth = sin(theta);

			g2o::VectorN<3, T> vXp = v.cross(point);
			T vDotp = v.dot(point);
			T oneMinusCth = T(1) - cth;

			p = point * cth + vXp * sth + v * vDotp * oneMinusCth;
		}
		else
		{
			// Taylor expansion for theta close to zero.
			p = point + camera.template head<3>().cross(point);
		}

		// Translation of the camera.
		p += camera.template segment<3>(3);

		// Perspective division.
		g2o::VectorN<2, T> projectedPoint = -p.template head<2>() / p(2);

		// Conversion to pixel coordinates.
		T radiusSqr = projectedPoint.squaredNorm();
		const T& f = camera(6);
		const T& k1 = camera(7);
		const T& k2 = camera(8);
		T r_p = T(1) + k1 * radiusSqr + k2 * radiusSqr * radiusSqr;
		g2o::VectorN<2, T> prediction = f * r_p * projectedPoint;

		// Compute the error.
		typename g2o::VectorN<2, T>::MapType error(p_error);
		error = prediction - measurement().cast<T>();
		(void)error;
		return true;
	}

	G2O_MAKE_AUTO_AD_FUNCTIONS
};

}  // namespace local
}  // unnamed namespace

namespace my_g2o {

// REF [site] >> https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/bal
void bal_example()
{
	const int maxIterations = 5;  // Perform n iterations.
	const bool verbose = false;  // Verbose output of the optimization process.
	const bool usePCG = false;  // Use PCG instead of the Cholesky.

	// REF [site] >> https://grail.cs.washington.edu/projects/bal/
	const std::string inputFilename("./problem-93-61203-pre.txt.bz2");  // File which will be processed.
	const std::string outputFilename("./bal.vrml");  // Write points into a vrml file.
	const std::string statsFilename("./bal_stat.txt");  // Specify a file for the statistics.

	typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BalBlockSolver;
#ifdef G2O_HAVE_CHOLMOD
	const std::string choleskySolverName = "CHOLMOD";
	typedef g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType> BalLinearSolver;
#else
	const std::string choleskySolverName = "Eigen";
	typedef g2o::LinearSolverEigen<BalBlockSolver::PoseMatrixType> BalLinearSolver;
#endif
	typedef g2o::LinearSolverPCG<BalBlockSolver::PoseMatrixType> BalLinearSolverPCG;

	g2o::SparseOptimizer optimizer;
	std::unique_ptr<g2o::LinearSolver<BalBlockSolver::PoseMatrixType>> linearSolver;
	if (usePCG)
	{
		std::cout << "Using PCG" << std::endl;
		linearSolver = g2o::make_unique<BalLinearSolverPCG>();
	}
	else
	{
		std::cout << "Using Cholesky: " << choleskySolverName << std::endl;
		auto cholesky = g2o::make_unique<BalLinearSolver>();
		cholesky->setBlockOrdering(true);
		linearSolver = std::move(cholesky);
	}
	auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BalBlockSolver>(std::move(linearSolver)));

	//solver->setUserLambdaInit(1);
	optimizer.setAlgorithm(solver);
	if (statsFilename.size() > 0)
	{
		optimizer.setComputeBatchStatistics(true);
	}

	std::vector<local::VertexPointBAL*> points;
	std::vector<local::VertexCameraBAL*> cameras;

	// Parse BAL dataset.
	std::cout << "Loading BAL dataset " << inputFilename << std::endl;
	{
		std::ifstream ifs(inputFilename.c_str());
		int numCameras, numPoints, numObservations;
		ifs >> numCameras >> numPoints >> numObservations;

		std::cerr << PVAR(numCameras) << " " << PVAR(numPoints) << " " << PVAR(numObservations) << std::endl;

		int id = 0;
		cameras.reserve(numCameras);
		for (int i = 0; i < numCameras; ++i, ++id)
		{
			auto cam = new local::VertexCameraBAL;
			cam->setId(id);
			optimizer.addVertex(cam);
			cameras.push_back(cam);
		}

		points.reserve(numPoints);
		for (int i = 0; i < numPoints; ++i, ++id)
		{
			auto p = new local::VertexPointBAL;
			p->setId(id);
			p->setMarginalized(true);
			const bool addedVertex = optimizer.addVertex(p);
			if (!addedVertex)
			{
				std::cerr << "failing adding vertex" << std::endl;
			}
			points.push_back(p);
		}

		// Read in the observation.
		for (int i = 0; i < numObservations; ++i)
		{
			int camIndex, pointIndex;
			double obsX, obsY;
			ifs >> camIndex >> pointIndex >> obsX >> obsY;

			assert(camIndex >= 0 && (size_t)camIndex < cameras.size() && "Index out of bounds");
			local::VertexCameraBAL* cam = cameras[camIndex];
			assert(pointIndex >= 0 && (size_t)pointIndex < points.size() && "Index out of bounds");
			local::VertexPointBAL* point = points[pointIndex];

			auto e = new local::EdgeObservationBAL;
			e->setVertex(0, cam);
			e->setVertex(1, point);
			e->setInformation(g2o::Matrix2::Identity());
			e->setMeasurement(g2o::Vector2(obsX, obsY));
			const bool addedEdge = optimizer.addEdge(e);
			if (!addedEdge)
			{
				std::cerr << "error adding edge" << std::endl;
			}
		}

		// Read in the camera params.
		for (int i = 0; i < numCameras; ++i)
		{
			g2o::bal::Vector9 cameraParameter;
			for (int j = 0; j < 9; ++j) ifs >> cameraParameter(j);
			local::VertexCameraBAL* cam = cameras[i];
			cam->setEstimate(cameraParameter);
		}

		// Read in the points.
		for (int i = 0; i < numPoints; ++i)
		{
			g2o::Vector3 p;
			ifs >> p(0) >> p(1) >> p(2);
			local::VertexPointBAL* point = points[i];
			point->setEstimate(p);
		}
	}
	std::cout << "done." << std::endl;

	std::cout << "Initializing ... " << std::flush;
	optimizer.initializeOptimization();
	std::cout << "done." << std::endl;
	optimizer.setVerbose(verbose);
	std::cout << "Start to optimize" << std::endl;
	optimizer.optimize(maxIterations);

	if (statsFilename != "")
	{
		std::cerr << "writing stats to file \"" << statsFilename << "\" ... ";
		std::ofstream fout(statsFilename.c_str());
		const g2o::BatchStatisticsContainer& bsc = optimizer.batchStatistics();
		for (size_t i = 0; i < bsc.size(); i++) fout << bsc[i] << std::endl;
		std::cerr << "done." << std::endl;
	}

	// Dump the points.
	if (outputFilename.size() > 0)
	{
		std::ofstream fout(outputFilename.c_str());  // Loadable with meshlab.
		fout << "#VRML V2.0 utf8\n"
			<< "Shape {\n"
			<< "  appearance Appearance {\n"
			<< "    material Material {\n"
			<< "      diffuseColor " << 1 << " " << 0 << " " << 0 << "\n"
			<< "      ambientIntensity 0.2\n"
			<< "      emissiveColor 0.0 0.0 0.0\n"
			<< "      specularColor 0.0 0.0 0.0\n"
			<< "      shininess 0.2\n"
			<< "      transparency 0.0\n"
			<< "    }\n"
			<< "  }\n"
			<< "  geometry PointSet {\n"
			<< "    coord Coordinate {\n"
			<< "      point [\n";
		for (std::vector<local::VertexPointBAL*>::const_iterator it = points.begin(); it != points.end(); ++it)
		{
			fout << (*it)->estimate().transpose() << std::endl;
		}
		fout << "    ]\n"
			<< "  }\n"
			<< "}\n"
			<< "  }\n";
	}
}

}  // namespace my_g2o
