//#include "stdafx.h"
#include <mrpt/core.h>
#include <cassert>


namespace {
namespace local {

void ransac_3d_plane_fit(const mrpt::math::CMatrixDouble &allData, const mrpt::vector_size_t &useIndices, std::vector<mrpt::math::CMatrixDouble> &fitModels)
{
	assert(useIndices.size() == 3);

	mrpt::math::TPoint3D p1(allData(0, useIndices[0]), allData(1, useIndices[0]), allData(2, useIndices[0]));
	mrpt::math::TPoint3D p2(allData(0, useIndices[1]), allData(1, useIndices[1]), allData(2, useIndices[1]));
	mrpt::math::TPoint3D p3(allData(0, useIndices[2]), allData(1, useIndices[2]), allData(2, useIndices[2]));

	try
	{
		mrpt::math::TPlane plane(p1, p2, p3);
		fitModels.resize(1);
		mrpt::math::CMatrixDouble &M = fitModels[0];

		M.setSize(1, 4);
		for (std::size_t i = 0; i < 4; ++i)
			M(0,i) = plane.coefs[i];
	}
	catch (const std::exception &)
	{
		fitModels.clear();
		return;
	}
}

void ransac_3d_plane_distance(const mrpt::math::CMatrixDouble &allData, const std::vector<mrpt::math::CMatrixDouble> &testModels, const double distanceThreshold, unsigned int &out_bestModelIndex, mrpt::vector_size_t &out_inlierIndices)
{
	assert(testModels.size() == 1);
	out_bestModelIndex = 0;
	const mrpt::math::CMatrixDouble &M = testModels[0];

	assert(mrpt::math::size(M,1) == 1 && mrpt::math::size(M,2) == 4);

	mrpt::math::TPlane plane;
	plane.coefs[0] = M(0,0);
	plane.coefs[1] = M(0,1);
	plane.coefs[2] = M(0,2);
	plane.coefs[3] = M(0,3);

	const std::size_t N = mrpt::math::size(allData, 2);
	out_inlierIndices.clear();
	out_inlierIndices.reserve(100);
	for (std::size_t i = 0; i < N; ++i)
	{
		const double d = plane.distance(mrpt::math::TPoint3D(allData.get_unsafe(0, i), allData.get_unsafe(1, i), allData.get_unsafe(2, i)));
		if (d < distanceThreshold)
			out_inlierIndices.push_back(i);
	}
}

// return "true" if the selected points are a degenerate (invalid) case.
bool ransac_3d_plane_degenerate(const mrpt::math::CMatrixDouble &allData, const mrpt::vector_size_t &useIndices)
{
	return false;
}

void ransac_3d_plane()
{
	mrpt::random::randomGenerator.randomize();

	//
	const std::size_t N_plane = 300;
	const std::size_t N_noise = 100;

	// generate random points:
	mrpt::math::CMatrixDouble data(3, N_plane + N_noise);
	{
		const double PLANE_EQ[4] = { 1, -1, 1, -2 };

		for (std::size_t i = 0; i < N_plane; ++i)
		{
			const double xx = mrpt::random::randomGenerator.drawUniform(-3, 3);
			const double yy = mrpt::random::randomGenerator.drawUniform(-3, 3);
			const double zz = -(PLANE_EQ[3] + PLANE_EQ[0] * xx + PLANE_EQ[1] * yy) / PLANE_EQ[2];
			data(0,i) = xx;
			data(1,i) = yy;
			data(2,i) = zz;
		}

		for (std::size_t i = 0; i < N_noise; ++i)
		{
			data(0,i+N_plane) = mrpt::random::randomGenerator.drawUniform(-4, 4);
			data(1,i+N_plane) = mrpt::random::randomGenerator.drawUniform(-4, 4);
			data(2,i+N_plane) = mrpt::random::randomGenerator.drawUniform(-4, 4);
		}
	}

	// run RANSAC
	mrpt::math::CMatrixDouble best_model;
	mrpt::vector_size_t best_inliers;
	const double DIST_THRESHOLD = 0.2;
	const std::size_t TIMES = 100;

	mrpt::utils::CTicTac tictac;

	for (std::size_t iters = 0; iters < TIMES; ++iters)
	{
		mrpt::math::RANSAC::execute(
			data,
			ransac_3d_plane_fit,
			ransac_3d_plane_distance,
			ransac_3d_plane_degenerate,
			DIST_THRESHOLD,
			3,  // minimum set of points
			best_inliers,
			best_model,
			iters == 0  // verbose
		);
	}

	std::cout << "computation time: " << (tictac.Tac() * 1000.0 / TIMES) << " ms" << std::endl;

	assert(mrpt::math::size(best_model, 1) == 1 && mrpt::math::size(best_model, 2) == 4);

	std::cout << "RANSAC finished: best model: " << best_model << std::endl;
	//std::cout << "best inliers: " << best_inliers << std::endl;

	// show GUI
	{
		mrpt::opengl::COpenGLScenePtr scene = mrpt::opengl::COpenGLScene::Create();

		scene->insert(mrpt::opengl::CGridPlaneXY::Create(-20, 20, -20, 20, 0, 1));
		scene->insert(mrpt::opengl::stock_objects::CornerXYZ());

		mrpt::opengl::CPointCloudPtr points = mrpt::opengl::CPointCloud::Create();
		points->getArrayX().resize(mrpt::math::size(data, 2));
		points->getArrayY().resize(mrpt::math::size(data, 2));
		points->getArrayZ().resize(mrpt::math::size(data, 2));
		points->setColor(0,0,1);
		points->setPointSize(3);
		points->enableColorFromZ();

		data.extractRow(0, points->getArrayX());
		data.extractRow(1, points->getArrayY());
		data.extractRow(2, points->getArrayZ());

		scene->insert(points);

		mrpt::opengl::CTexturedPlanePtr glPlane = mrpt::opengl::CTexturedPlane::Create(-4, 4, -4, 4);

		mrpt::math::TPlane plane(best_model(0, 0), best_model(0, 1), best_model(0, 2), best_model(0, 3));

		mrpt::poses::CPose3D glPlanePose;
		plane.getAsPose3D(glPlanePose);
		glPlane->setPose(glPlanePose);

		scene->insert(glPlane);

		mrpt::gui::CDisplayWindow3D win("set of points", 500, 500);
		win.get3DSceneAndLock() = scene;
		win.unlockAccess3DScene();
		win.forceRepaint();

		win.waitForKey();
	}
}

void ransac_planes()
{
	mrpt::random::randomGenerator.randomize();

	// generate random points:
	mrpt::vector_double xs, ys, zs;
	{
		const std::size_t N_PLANES = 3;

		const std::size_t N_plane = 300;
		const std::size_t N_noise = 300;

		const double PLANE_EQ[N_PLANES][4] = { 
			{ 1, -1, 1, -2 },
			{ 1, +1.5, 1, -1 },
			{ 0, -1, 1, +2 }
		};

		for (std::size_t pl = 0; pl < N_PLANES; ++pl)
		{
			for (std::size_t i = 0; i < N_plane; ++i)
			{
				const double xx = mrpt::random::randomGenerator.drawUniform(-3, 3) + 5 * std::cos(0.4 * pl);
				const double yy = mrpt::random::randomGenerator.drawUniform(-3, 3) + 5 * std::sin(0.4 * pl);
				const double zz = -(PLANE_EQ[pl][3] + PLANE_EQ[pl][0] * xx + PLANE_EQ[pl][1] * yy) / PLANE_EQ[pl][2];
				xs.push_back(xx);
				ys.push_back(yy);
				zs.push_back(zz);
			}
		}

		for (std::size_t i = 0; i < N_noise; ++i)
		{
			xs.push_back(mrpt::random::randomGenerator.drawUniform(-7, 7));
			ys.push_back(mrpt::random::randomGenerator.drawUniform(-7, 7));
			zs.push_back(mrpt::random::randomGenerator.drawUniform(-7, 7));
		}
	}

	// run RANSAC
	std::vector<std::pair<std::size_t, mrpt::math::TPlane> > detectedPlanes;
	const double DIST_THRESHOLD = 0.05;

	mrpt::utils::CTicTac tictac;

	mrpt::math::ransac_detect_3D_planes(xs, ys, zs, detectedPlanes, DIST_THRESHOLD, 40);

	// display output:
	std::cout << "RANSAC method: ransac_detect_3D_planes" << std::endl;
	std::cout << " computation time: " << tictac.Tac() * 1000.0 << " ms" << std::endl;
	std::cout << " " << detectedPlanes.size() << " planes detected." << std::endl;

	// show GUI
	{
		mrpt::opengl::COpenGLScenePtr scene = mrpt::opengl::COpenGLScene::Create();

		scene->insert(mrpt::opengl::CGridPlaneXY::Create(-20, 20, -20, 20, 0, 1));
		scene->insert(mrpt::opengl::stock_objects::CornerXYZ());

		for (std::vector<std::pair<std::size_t, mrpt::math::TPlane> >::iterator it = detectedPlanes.begin(); it != detectedPlanes.end(); ++it)
		{
			mrpt::opengl::CTexturedPlanePtr glPlane = mrpt::opengl::CTexturedPlane::Create(-10, 10, -10, 10);

			mrpt::poses::CPose3D glPlanePose;
			it->second.getAsPose3D(glPlanePose);
			glPlane->setPose(glPlanePose);

			glPlane->setColor(mrpt::random::randomGenerator.drawUniform(0, 1), mrpt::random::randomGenerator.drawUniform(0, 1), mrpt::random::randomGenerator.drawUniform(0, 1), 0.6);

			scene->insert(glPlane);
		}

		{
			mrpt::opengl::CPointCloudPtr points = mrpt::opengl::CPointCloud::Create();
			points->setColor(0, 0, 1);
			points->setPointSize(3);
			points->enableColorFromZ();

			mrpt::utils::metaprogramming::copy_container_typecasting(xs, points->getArrayX());
			mrpt::utils::metaprogramming::copy_container_typecasting(ys, points->getArrayY());
			mrpt::utils::metaprogramming::copy_container_typecasting(zs, points->getArrayZ());

			scene->insert(points);
		}

		mrpt::gui::CDisplayWindow3D win("RANSAC: 3D planes", 500, 500);
		win.get3DSceneAndLock() = scene;
		win.unlockAccess3DScene();
		win.forceRepaint();

		win.waitForKey();
	}
}

void ransac_lines()
{
	mrpt::random::randomGenerator.randomize();

	// generate random points in 2D
	mrpt::vector_double xs, ys;
	{
		const std::size_t N_LINES = 4;

		const std::size_t N_line = 30;
		const std::size_t N_noise = 50;

		const double LINE_EQ[N_LINES][3] = { 
			{ 1, -1, -2 },
			{ 1, +1.5, -1 },
			{ 0, -1, +2 },
			{ 0.5, -0.3, +1 }
		};

		for (std::size_t ln = 0; ln < N_LINES; ++ln)
		{
			for (std::size_t i = 0; i < N_line; ++i)
			{
				const double xx = mrpt::random::randomGenerator.drawUniform(-10, 10);
				const double yy = mrpt::random::randomGenerator.drawGaussian1D(0, 0.05) - (LINE_EQ[ln][2] + LINE_EQ[ln][0] * xx) / LINE_EQ[ln][1];
				xs.push_back(xx);
				ys.push_back(yy);
			}
		}

		for (std::size_t i = 0; i < N_noise; ++i)
		{
			xs.push_back(mrpt::random::randomGenerator.drawUniform(-15, 15));
			ys.push_back(mrpt::random::randomGenerator.drawUniform(-15, 15));
		}
	}

	// run RANSAC
	std::vector<std::pair<std::size_t, mrpt::math::TLine2D> > detectedLines;
	const double DIST_THRESHOLD = 0.2;

	mrpt::utils::CTicTac tictac;

	mrpt::math::ransac_detect_2D_lines(xs, ys, detectedLines, DIST_THRESHOLD, 20);

	// display output:
	std::cout << "RANSAC method: ransac_detect_2D_lines" << std::endl;
	std::cout << " Computation time: " << tictac.Tac() * 1000.0 << " ms" << std::endl;
	std::cout << " " << detectedLines.size() << " lines detected." << std::endl;

	// show GUI
	{
		mrpt::gui::CDisplayWindowPlots win("set of points", 500, 500);

		win.plot(xs, ys, ".b4", "points");

		unsigned int n = 0;
		for (std::vector<std::pair<std::size_t, mrpt::math::TLine2D> >::iterator it = detectedLines.begin(); it != detectedLines.end(); ++it)
		{
			mrpt::vector_double lx(2), ly(2);
			lx[0] = -15;
			lx[1] = 15;
			for (std::size_t q = 0; q < lx.size(); ++q)
				ly[q] = -(it->second.coefs[2] + it->second.coefs[0] * lx[q]) / it->second.coefs[1];
			win.plot(lx, ly, "r-1", mrpt::format("line_%u", n++));
		}

		win.axis_fit();
		win.axis_equal();

		win.waitForKey();
	}
}

}  // namespace local
}  // unnamed namespace

void ransac()
{
	//local::ransac_3d_plane();

	local::ransac_planes();
	//local::ransac_lines();
}