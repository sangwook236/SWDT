#include <chrono>
#include <string>
#include <iostream>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_geometry_handlers.h>
#include <pcl/visualization/point_cloud_color_handlers.h>


namespace {
namespace local {

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/tools/marching_cubes_reconstruction.cpp
void marching_cubes_reconstruction_example()
{
	const float iso_level = 0.0f;  // The iso level of the surface to be extracted
	const int hoppe_or_rbf = 0;  // Use the Hoppe or RBF signed distance function (MarchingCubesHoppe or MarchingCubesRBF)
	const float extend_percentage = 0.0f;  // The percentage of the bounding box to extend the grid by
	const int grid_res = 50;  // The resolution of the grid (cubic grid)
	const float off_surface_displacement = 0.01f;  // The displacement value for the off-surface points (only for RBF)

	const std::string pcd_filepath("./input.pcd");
	const std::string vtk_filepath("./mc_output.vtk");

	pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2);
	{
		std::cout << "Loading a point cloud from " << pcd_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//if (pcl::io::loadPCDFile(pcd_filepath, *cloud) < 0)
		pcl::PCDReader reader;
		if (reader.read(pcd_filepath, *cloud) != 0)
		{
			std::cerr << "A point cloud file not found, " << pcd_filepath << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "A point cloud loaded: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud->width * cloud->height << " points." << std::endl;
		std::cout << "Available dimensions: " << pcl::getFieldsList(*cloud).c_str() << std::endl;
	}

	pcl::PolygonMesh polygon_mesh;
	{
		// Bilateral filtering
		//	Refer to KinectFusion
		// Mean least squres (MLS) smoothing.
		//	Refer to mls_smoothing_example()

#if 1
		pcl::MarchingCubes<pcl::PointNormal> *mc = nullptr;
		if (0 == hoppe_or_rbf)
			mc = new pcl::MarchingCubesHoppe<pcl::PointNormal>();
		else
		{
			mc = new pcl::MarchingCubesRBF<pcl::PointNormal>();
			(reinterpret_cast<pcl::MarchingCubesRBF<pcl::PointNormal>*>(mc))->setOffSurfaceDisplacement(off_surface_displacement);
		}
#else
		std::unique_ptr<pcl::MarchingCubes<pcl::PointNormal> > mc;
		if (0 == hoppe_or_rbf)
			mc.reset(new pcl::MarchingCubesHoppe<pcl::PointNormal>());
		else
		{
			mc.reset(new pcl::MarchingCubesRBF<pcl::PointNormal>());
			(reinterpret_cast<pcl::MarchingCubesRBF<pcl::PointNormal>*>(mc.get()))->setOffSurfaceDisplacement(off_surface_displacement);
		}
#endif

		pcl::PointCloud<pcl::PointNormal>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointNormal>());
#if 0
		pcl::fromPCLPointCloud2(*cloud, *xyz_cloud);
#else
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::fromPCLPointCloud2(*cloud, *cloud_xyz);

		//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setInputCloud(cloud_xyz);
		//ne.setNumberOfThreads(8);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(0.03);

		pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
		ne.compute(*cloud_normals);

		pcl::concatenateFields(*cloud_xyz, *cloud_normals, *xyz_cloud);
#endif

		mc->setIsoLevel(iso_level);
		mc->setGridResolution(grid_res, grid_res, grid_res);
		mc->setPercentageExtendGrid(extend_percentage);
		mc->setInputCloud(xyz_cloud);

		std::cout << "Reconstructing surface..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		mc->reconstruct(polygon_mesh);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Surface reconstructed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

		// NOTE [error] >> Segmentation fault (core dumped)
		//	When deleting the instance, mc
		delete mc;
		mc = nullptr;
	}

	{
		std::cout << "Saving the reconstructed surface to " << vtk_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		if (pcl::io::saveVTKFile(vtk_filepath, polygon_mesh, /*precision =*/ 5) < 0)
		{
			std::cerr << "The reconstructed surface not saved to " << vtk_filepath << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "The reconstructed surface saved: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}


#if 0
	// Not working

	// Visualize.
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	viewer.addPolygonMesh(polygon_mesh, "polygon mesh");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
#endif
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/tools/poisson_reconstruction.cpp
void poisson_reconstruction_example()
{
	const int depth = 8;  // The maximum depth of the tree that will be used for surface reconstruction
	const int solver_divide = 8;  // The the depth at which a block Gauss-Seidel solver is used to solve the Laplacian equation
	const int iso_divide = 8;  // The depth at which a block iso-surface extractor should be used to extract the iso-surface
	const float point_weight = 4.0f;  // The importance that interpolation of the point samples is given in the formulation of the screened Poisson equation. The results of the original (unscreened) Poisson Reconstruction can be obtained by setting this value to 0

	const std::string pcd_filepath("./input.pcd");
	const std::string vtk_filepath("./psr_output.vtk");

	pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2);
	{
		std::cout << "Loading a point cloud from " << pcd_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//if (pcl::io::loadPCDFile(pcd_filepath, *cloud) < 0)
		pcl::PCDReader reader;
		if (reader.read(pcd_filepath, *cloud) != 0)
		{
			std::cerr << "A point cloud file not found, " << pcd_filepath << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "A point cloud loaded: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud->width * cloud->height << " points." << std::endl;
		std::cout << "Available dimensions: " << pcl::getFieldsList(*cloud).c_str() << std::endl;
	}

	pcl::PolygonMesh polygon_mesh;
	{
		std::cout << "Using parameters: depth " << depth <<", solverDivide " << solver_divide << ", isoDivide " << iso_divide << std::endl;

		// Bilateral filtering
		//	Refer to KinectFusion
		// Mean least squres (MLS) smoothing.
		//	Refer to mls_smoothing_example()

		pcl::PointCloud<pcl::PointNormal>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointNormal>());
#if 0
		pcl::fromPCLPointCloud2(*cloud, *xyz_cloud);
#else
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::fromPCLPointCloud2(*cloud, *cloud_xyz);

		//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setInputCloud(cloud_xyz);
		//ne.setNumberOfThreads(8);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(0.03);

		pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
		ne.compute(*cloud_normals);

		pcl::concatenateFields(*cloud_xyz, *cloud_normals, *xyz_cloud);
#endif

		pcl::Poisson<pcl::PointNormal> poisson;
		poisson.setDepth(depth);
		poisson.setSolverDivide(solver_divide);
		poisson.setIsoDivide(iso_divide);
		poisson.setPointWeight(point_weight);
		poisson.setInputCloud(xyz_cloud);

		std::cout << "Reconstructing surface..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		poisson.reconstruct(polygon_mesh);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Surface reconstructed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}

	{
		std::cout << "Saving the reconstructed surface to " << vtk_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		if (pcl::io::saveVTKFile(vtk_filepath, polygon_mesh, /*precision =*/ 5) < 0)
		{
			std::cerr << "The reconstructed surface not saved to " << vtk_filepath << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "The reconstructed surface saved: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}

#if 0
	// Not working

	// Visualize.
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	viewer.addPolygonMesh(polygon_mesh, "polygon mesh");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
#endif
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/tools/mls_smoothing.cpp
void mls_smoothing_example()
{
	const double search_radius = 0.01;  // Sphere radius to be used for finding the k-nearest neighbors used for fitting
	const double sqr_gauss_param = search_radius * search_radius;  // Parameter used for the distance based weighting of neighbors (recommended = search_radius^2)
	const bool sqr_gauss_param_set = true;
	const int polynomial_order = 2;  // Order of the polynomial to be fit (polynomial_order > 1, indicates using a polynomial fit)

	const std::string input_filepath("./input.pcd");
	const std::string output_filepath("./mls_output.pcd");

	pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2);
	{
		std::cout << "Loading a point cloud from " << input_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//if (pcl::io::loadPCDFile(input_filepath, *cloud) < 0)
		pcl::PCDReader reader;
		if (reader.read(input_filepath, *cloud) != 0)
		{
			std::cerr << "A point cloud file not found, " << input_filepath << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "A point cloud loaded: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud->width * cloud->height << " points." << std::endl;
		std::cout << "Available dimensions: " << pcl::getFieldsList(*cloud).c_str() << std::endl;
	}

	// Do the smoothing
	pcl::PCLPointCloud2 cloud_smoothed;
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud_pre(new pcl::PointCloud<pcl::PointXYZ>()), xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::fromPCLPointCloud2(*cloud, *xyz_cloud_pre);

		// Filter the NaNs from the cloud
		for (std::size_t i = 0; i < xyz_cloud_pre->size(); ++i)
			if (std::isfinite((*xyz_cloud_pre)[i].x))
				xyz_cloud->push_back((*xyz_cloud_pre)[i]);
		xyz_cloud->header = xyz_cloud_pre->header;
		xyz_cloud->height = 1;
		xyz_cloud->width = xyz_cloud->size();
		xyz_cloud->is_dense = false;

		pcl::PointCloud<pcl::PointNormal>::Ptr xyz_cloud_smoothed(new pcl::PointCloud<pcl::PointNormal>());

		pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
		mls.setInputCloud(xyz_cloud);
		mls.setSearchRadius(search_radius);
		if (sqr_gauss_param_set) mls.setSqrGaussParam(sqr_gauss_param);
		mls.setPolynomialOrder(polynomial_order);

		//mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::SAMPLE_LOCAL_PLANE);
		//mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::RANDOM_UNIFORM_DENSITY);
		//mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::VOXEL_GRID_DILATION);
		mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::NONE);
		mls.setPointDensity(60000 * static_cast<int>(search_radius));  // 300 points in a 5 cm radius
		mls.setUpsamplingRadius(0.025);
		mls.setUpsamplingStepSize(0.015);
		mls.setDilationIterations(2);
		mls.setDilationVoxelSize(0.01f);

		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		mls.setSearchMethod(tree);
		mls.setComputeNormals(true);

		std::cout << "Computing smoothed surface and normals with search_radius " << mls.getSearchRadius() << ", sqr_gaussian_param " << mls.getSqrGaussParam() << ", polynomial order " << mls.getPolynomialOrder() << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		mls.process(*xyz_cloud_smoothed);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Smoothed surface and normals computed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << xyz_cloud_smoothed->width * xyz_cloud_smoothed->height << " points smoothed." << std::endl;

		pcl::toPCLPointCloud2(*xyz_cloud_smoothed, cloud_smoothed);
	}

	{
		std::cout << "Saving the smoothed point cloud to " << output_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//if (pcl::io::savePCDFile(output_filepath, cloud_smoothed, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), true) < 0)
		pcl::PCDWriter writer;
		if (writer.write(output_filepath, cloud_smoothed) != 0)
		{
			std::cerr << "The smoothed point cloud not saved." << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "The smoothed point cloud saved: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}

#if 0
	// Not working

	// Visualize.
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	pcl::visualization::PointCloudGeometryHandlerCustom<pcl::PCLPointCloud2> geometry_handler(pcl::PCLPointCloud2::Ptr(&cloud_smoothed), "x", "y", "z");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PCLPointCloud2> color_handler(pcl::PCLPointCloud2::Ptr(&cloud_smoothed), 150, 150, 150);
	viewer.addPointCloud(
		pcl::PCLPointCloud2::Ptr(&cloud_smoothed),
		pcl::visualization::PointCloudGeometryHandler<pcl::PCLPointCloud2>::Ptr(&geometry_handler), pcl::visualization::PointCloudColorHandler<pcl::PCLPointCloud2>::Ptr(&color_handler),
		Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(),
		"point cloud"
	);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void reconstruction()
{
	// Surface reconstruction

	// Marching cubes
	//local::marching_cubes_reconstruction_example();

	// (screened) Poisson surface reconstruction (PSR or SPSR)
	local::poisson_reconstruction_example();

	// Moving least squares (MLS)
	//	An algorithm for data smoothing and improved normal estimation
	//local::mls_smoothing_example();
}

}  // namespace my_pcl
