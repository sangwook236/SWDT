#include <chrono>
#include <string>
#include <iostream>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/fast_bilateral_omp.h>
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
	const std::string pcd_filepath("./input.pcd");
	const std::string vtk_filepath("./mc_output.vtk");

	// Load a point cloud from a file
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

#if 0
	// Bilateral filtering
	//	Refer to KinectFusion
	//if (cloud->width > 1 and cloud->height > 1)
	if (cloud->height > 1)
	//if (cloud->isOrganized())
	{
		// FIXME [implement] >>
	}
#endif

	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn(new pcl::PointCloud<pcl::PointNormal>());
#if 0
	{
		// Moving least squares (MLS) smoothing
		//	REF [function] >> mls_smoothing_example()

		const double search_radius = 0.01;  // Sphere radius to be used for finding the k-nearest neighbors used for fitting
		const double sqr_gauss_param = search_radius * search_radius;  // Parameter used for the distance based weighting of neighbors (recommended = search_radius^2)
		const bool sqr_gauss_param_set = true;
		const int polynomial_order = 2;  // Order of the polynomial to be fit (polynomial_order > 1, indicates using a polynomial fit)

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());

		// Filter the NaNs from the cloud
		for (std::size_t i = 0; i < cloud->size(); ++i)
			if (std::isfinite((*cloud)[i].x))
				cloud_xyz->push_back((*cloud)[i]);
		cloud_xyz->header = cloud->header;
		cloud_xyz->height = 1;
		cloud_xyz->width = cloud_xyz->size();
		cloud_xyz->is_dense = false;

		pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
		mls.setInputCloud(cloud_xyz);
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
		mls.process(*cloud_pn);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Smoothed surface and normals computed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud_pn->width * cloud_pn->height << " points smoothed." << std::endl;
	}
#elif 1
	{
		// Normal estimation

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::fromPCLPointCloud2(*cloud, *cloud_xyz);

		//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setInputCloud(cloud_xyz);
		//ne.setNumberOfThreads(8);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(0.03);

		pcl::PointCloud<pcl::Normal>::Ptr cloud_normal(new pcl::PointCloud<pcl::Normal>());
		ne.compute(*cloud_normal);

		pcl::concatenateFields(*cloud_xyz, *cloud_normal, *cloud_pn);
	}
#else
	pcl::fromPCLPointCloud2(*cloud, *cloud_pn);
#endif

	// Surface reconstruction
	pcl::PolygonMesh polygon_mesh;
	{
		const float iso_level = 0.0f;  // The iso level of the surface to be extracted
		const int hoppe_or_rbf = 0;  // Use the Hoppe or RBF signed distance function (MarchingCubesHoppe or MarchingCubesRBF)
		const float extend_percentage = 0.0f;  // The percentage of the bounding box to extend the grid by
		const int grid_res = 50;  // The resolution of the grid (cubic grid)
		const float off_surface_displacement = 0.01f;  // The displacement value for the off-surface points (only for RBF)

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

		mc->setIsoLevel(iso_level);
		mc->setGridResolution(grid_res, grid_res, grid_res);
		mc->setPercentageExtendGrid(extend_percentage);
		mc->setInputCloud(cloud_pn);

		std::cout << "Reconstructing surface..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		mc->reconstruct(polygon_mesh);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Surface reconstructed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << "#polygons reconstructed = " <<polygon_mesh.polygons.size() << std::endl;

		// NOTE [error] >> Segmentation fault (core dumped)
		//	When deleting the instance, mc
		delete mc;
		mc = nullptr;
	}

	// Save the reconstructed surface to a file.
	{
		std::cout << "Saving the reconstructed surface to " << vtk_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		if (pcl::io::saveVTKFile(vtk_filepath, polygon_mesh, /*precision =*/ 5) < 0)
		{
			std::cerr << "The reconstructed surface not saved." << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "The reconstructed surface saved: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}


#if 0
	// Not working

	// Visualize
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	viewer.addPolygonMesh(polygon_mesh, "polygon mesh");
#if 0
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler_cloud(cloud, 255, 0, 0);
	//viewer.addPointCloud<pcl::PointXYZ>(cloud, color_handler_cloud, "point cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> color_handler_cloud(cloud_pn, 255, 0, 0);
	viewer.addPointCloud<pcl::PointNormal>(cloud_pn, color_handler_cloud, "point cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "point cloud");
#endif
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
#endif
}

// REF [function] >> marching_cubes_reconstruction_example()
void marching_cubes_reconstruction_test()
{
	const std::string pcd_filepath("./input.pcd");
	const std::string vtk_filepath("./mc_output.vtk");

	// Load a point cloud from a file
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
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

#if 0
	// Bilateral filtering
	//	REF [function] >> fast_bilateral_filter_test() in ./pcl_filtering.cpp
	//	Refer to KinectFusion
	if (cloud->isOrganized())  // cloud->height != 1
	{
		const float sigma_s = 2.0f;  // The standard deviation of the Gaussian used by the bilateral filter for the spatial neighborhood/window
		const float sigma_r = 0.01f;  // The standard deviation of the Gaussian used to control how much an adjacent pixel is downweighted because of the intensity difference (depth in our case)

		//pcl::FastBilateralFilter<pcl::PointXYZ> fbf;
		pcl::FastBilateralFilterOMP<pcl::PointXYZ> fbf;
		fbf.setInputCloud(cloud);
		fbf.setSigmaS(sigma_s);
		fbf.setSigmaR(sigma_r);

		std::cout << "Filtering..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		fbf.filter(*cloud);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Filtered: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud->width * cloud->height << " points filtered." << std::endl;
	}
	else
	{
		std::cerr << "The point cloud is unorganized: width = " << cloud->width << ", height = " << cloud->height << std::endl;
		//return;
	}
#endif

#if 0
	// Moving least squares (MLS) smoothing
	//	REF [function] >> mls_smoothing_example()
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn(new pcl::PointCloud<pcl::PointNormal>());
	{
		const double search_radius = 0.01;  // Sphere radius to be used for finding the k-nearest neighbors used for fitting
		const double sqr_gauss_param = search_radius * search_radius;  // Parameter used for the distance based weighting of neighbors (recommended = search_radius^2)
		const bool sqr_gauss_param_set = true;
		const int polynomial_order = 2;  // Order of the polynomial to be fit (polynomial_order > 1, indicates using a polynomial fit)

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());

		// Filter the NaNs from the cloud
		for (std::size_t i = 0; i < cloud->size(); ++i)
			if (std::isfinite((*cloud)[i].x))
				cloud_xyz->push_back((*cloud)[i]);
		cloud_xyz->header = cloud->header;
		cloud_xyz->height = 1;
		cloud_xyz->width = cloud_xyz->size();
		cloud_xyz->is_dense = false;

		pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
		mls.setInputCloud(cloud_xyz);
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
		mls.process(*cloud_pn);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Smoothed surface and normals computed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud_pn->width * cloud_pn->height << " points smoothed." << std::endl;
	}
#elif 1
	// Normal estimation
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn(new pcl::PointCloud<pcl::PointNormal>());
	{
		//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setInputCloud(cloud);
		//ne.setNumberOfThreads(8);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(0.03);

		pcl::PointCloud<pcl::Normal>::Ptr cloud_normal(new pcl::PointCloud<pcl::Normal>());
		ne.compute(*cloud_normal);

		pcl::concatenateFields(*cloud, *cloud_normal, *cloud_pn);
	}
#else
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn = cloud;
#endif

	// Surface reconstruction
	pcl::PolygonMesh polygon_mesh;
	{
		const float iso_level = 0.0f;  // The iso level of the surface to be extracted
		const int hoppe_or_rbf = 0;  // Use the Hoppe or RBF signed distance function (MarchingCubesHoppe or MarchingCubesRBF)
		const float extend_percentage = 0.0f;  // The percentage of the bounding box to extend the grid by
		const int grid_res = 50;  // The resolution of the grid (cubic grid)
		const float off_surface_displacement = 0.01f;  // The displacement value for the off-surface points (only for RBF)

#if 0
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

		mc->setIsoLevel(iso_level);
		mc->setGridResolution(grid_res, grid_res, grid_res);
		mc->setPercentageExtendGrid(extend_percentage);
		mc->setInputCloud(cloud_pn);

		std::cout << "Reconstructing surface..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		mc->reconstruct(polygon_mesh);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Surface reconstructed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << "#polygons reconstructed = " <<polygon_mesh.polygons.size() << std::endl;

#if 0
		delete mc;
		mc = nullptr;
#endif
	}

	// Save the reconstructed surface to a file.
	{
		std::cout << "Saving the reconstructed surface to " << vtk_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		if (pcl::io::saveVTKFile(vtk_filepath, polygon_mesh, /*precision =*/ 5) < 0)
		{
			std::cerr << "The reconstructed surface not saved." << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "The reconstructed surface saved: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}


#if 1
	// Visualize
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	viewer.addPolygonMesh(polygon_mesh, "polygon mesh");
#if 0
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler_cloud(cloud, 255, 0, 0);
	//viewer.addPointCloud<pcl::PointXYZ>(cloud, color_handler_cloud, "point cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> color_handler_cloud(cloud_pn, 255, 0, 0);
	viewer.addPointCloud<pcl::PointNormal>(cloud_pn, color_handler_cloud, "point cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "point cloud");
#endif
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
#endif
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/tools/poisson_reconstruction.cpp
void poisson_reconstruction_example()
{
	const std::string pcd_filepath("./input.pcd");
	const std::string vtk_filepath("./psr_output.vtk");

	// Load a point cloud from a file
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

#if 0
	// Bilateral filtering
	//	Refer to KinectFusion
	if (cloud->height > 1)
	//if (cloud->isOrganized())
	{
		// FIXME [implement] >>
	}
#endif

	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn(new pcl::PointCloud<pcl::PointNormal>());
#if 0
	{
		// Moving least squares (MLS) smoothing
		//	REF [function] >> mls_smoothing_example()

		const double search_radius = 0.01;  // Sphere radius to be used for finding the k-nearest neighbors used for fitting
		const double sqr_gauss_param = search_radius * search_radius;  // Parameter used for the distance based weighting of neighbors (recommended = search_radius^2)
		const bool sqr_gauss_param_set = true;
		const int polynomial_order = 2;  // Order of the polynomial to be fit (polynomial_order > 1, indicates using a polynomial fit)

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());

		// Filter the NaNs from the cloud
		for (std::size_t i = 0; i < cloud->size(); ++i)
			if (std::isfinite((*cloud)[i].x))
				cloud_xyz->push_back((*cloud)[i]);
		cloud_xyz->header = cloud->header;
		cloud_xyz->height = 1;
		cloud_xyz->width = cloud_xyz->size();
		cloud_xyz->is_dense = false;

		pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
		mls.setInputCloud(cloud_xyz);
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
		mls.process(*cloud_pn);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Smoothed surface and normals computed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud_pn->width * cloud_pn->height << " points smoothed." << std::endl;
	}
#elif 1
	{
		// Normal estimation

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::fromPCLPointCloud2(*cloud, *cloud_xyz);

		//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setInputCloud(cloud_xyz);
		//ne.setNumberOfThreads(8);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(0.03);

		pcl::PointCloud<pcl::Normal>::Ptr cloud_normal(new pcl::PointCloud<pcl::Normal>());
		ne.compute(*cloud_normal);

		pcl::concatenateFields(*cloud_xyz, *cloud_normal, *cloud_pn);
	}
#else
	pcl::fromPCLPointCloud2(*cloud, *cloud_pn);
#endif

	// Surface reconstruction
	pcl::PolygonMesh polygon_mesh;
	{
		const int depth = 8;  // The maximum depth of the tree that will be used for surface reconstruction
		const int solver_divide = 8;  // The the depth at which a block Gauss-Seidel solver is used to solve the Laplacian equation
		const int iso_divide = 8;  // The depth at which a block iso-surface extractor should be used to extract the iso-surface
		const float point_weight = 4.0f;  // The importance that interpolation of the point samples is given in the formulation of the screened Poisson equation. The results of the original (unscreened) Poisson Reconstruction can be obtained by setting this value to 0

		std::cout << "Using parameters: depth = " << depth <<", solverDivide = " << solver_divide << ", isoDivide = " << iso_divide << std::endl;

		pcl::Poisson<pcl::PointNormal> poisson;
		poisson.setDepth(depth);
		poisson.setSolverDivide(solver_divide);
		poisson.setIsoDivide(iso_divide);
		poisson.setPointWeight(point_weight);
		poisson.setInputCloud(cloud_pn);

		std::cout << "Reconstructing surface..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		poisson.reconstruct(polygon_mesh);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Surface reconstructed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << "#polygons reconstructed = " <<polygon_mesh.polygons.size() << std::endl;
	}

	// Save the reconstructed surface to a file.
	{
		std::cout << "Saving the reconstructed surface to " << vtk_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		if (pcl::io::saveVTKFile(vtk_filepath, polygon_mesh, /*precision =*/ 5) < 0)
		{
			std::cerr << "The reconstructed surface not saved." << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "The reconstructed surface saved: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}

#if 0
	// Not working

	// Visualize
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	viewer.addPolygonMesh(polygon_mesh, "polygon mesh");
#if 0
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler_cloud(cloud, 255, 0, 0);
	//viewer.addPointCloud<pcl::PointXYZ>(cloud, color_handler_cloud, "point cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> color_handler_cloud(cloud_pn, 255, 0, 0);
	viewer.addPointCloud<pcl::PointNormal>(cloud_pn, color_handler_cloud, "point cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "point cloud");
#endif
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
#endif
}

// REF [function] >> poisson_reconstruction_example()
void poisson_reconstruction_test()
{
	const std::string pcd_filepath("./input.pcd");
	const std::string vtk_filepath("./psr_output.vtk");

	// Load a point cloud from a file
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
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

#if 0
	// Bilateral filtering
	//	REF [function] >> fast_bilateral_filter_test() in ./pcl_filtering.cpp
	//	Refer to KinectFusion
	if (cloud->isOrganized())  // cloud->height != 1
	{
		const float sigma_s = 2.0f;  // The standard deviation of the Gaussian used by the bilateral filter for the spatial neighborhood/window
		const float sigma_r = 0.01f;  // The standard deviation of the Gaussian used to control how much an adjacent pixel is downweighted because of the intensity difference (depth in our case)

		pcl::FastBilateralFilter<pcl::PointXYZ> fbf;
		//pcl::FastBilateralFilterOMP<pcl::PointXYZ> fbf;
		fbf.setInputCloud(cloud);
		fbf.setSigmaS(sigma_s);
		fbf.setSigmaR(sigma_r);

		std::cout << "Filtering..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		fbf.filter(*cloud);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Filtered: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud->width * cloud->height << " points filtered." << std::endl;
	}
	else
	{
		std::cerr << "The point cloud is unorganized: width = " << cloud->width << ", height = " << cloud->height << std::endl;
		//return;
	}
#endif

#if 0
	// Moving least squares (MLS) smoothing
	//	REF [function] >> mls_smoothing_example()
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn(new pcl::PointCloud<pcl::PointNormal>());
	{
		const double search_radius = 0.01;  // Sphere radius to be used for finding the k-nearest neighbors used for fitting
		const double sqr_gauss_param = search_radius * search_radius;  // Parameter used for the distance based weighting of neighbors (recommended = search_radius^2)
		const bool sqr_gauss_param_set = true;
		const int polynomial_order = 2;  // Order of the polynomial to be fit (polynomial_order > 1, indicates using a polynomial fit)

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());

		// Filter the NaNs from the cloud
		for (std::size_t i = 0; i < cloud->size(); ++i)
			if (std::isfinite((*cloud)[i].x))
				cloud_xyz->push_back((*cloud)[i]);
		cloud_xyz->header = cloud->header;
		cloud_xyz->height = 1;
		cloud_xyz->width = cloud_xyz->size();
		cloud_xyz->is_dense = false;

		pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
		mls.setInputCloud(cloud_xyz);
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
		mls.process(*cloud_pn);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Smoothed surface and normals computed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud_pn->width * cloud_pn->height << " points smoothed." << std::endl;
	}
#elif 1
	// Normal estimation
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn(new pcl::PointCloud<pcl::PointNormal>());
	{
		//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setInputCloud(cloud);
		//ne.setNumberOfThreads(8);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(0.03);

		pcl::PointCloud<pcl::Normal>::Ptr cloud_normal(new pcl::PointCloud<pcl::Normal>());
		ne.compute(*cloud_normal);

		pcl::concatenateFields(*cloud, *cloud_normal, *cloud_pn);
	}
#else
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn = cloud;
#endif

	// Surface reconstruction
	pcl::PolygonMesh polygon_mesh;
	{
		const int depth = 8;  // The maximum depth of the tree that will be used for surface reconstruction
		const int solver_divide = 8;  // The the depth at which a block Gauss-Seidel solver is used to solve the Laplacian equation
		const int iso_divide = 8;  // The depth at which a block iso-surface extractor should be used to extract the iso-surface
		const float point_weight = 4.0f;  // The importance that interpolation of the point samples is given in the formulation of the screened Poisson equation. The results of the original (unscreened) Poisson Reconstruction can be obtained by setting this value to 0

		std::cout << "Using parameters: depth = " << depth <<", solverDivide = " << solver_divide << ", isoDivide = " << iso_divide << std::endl;

		pcl::Poisson<pcl::PointNormal> poisson;
		poisson.setDepth(depth);
		poisson.setSolverDivide(solver_divide);
		poisson.setIsoDivide(iso_divide);
		poisson.setPointWeight(point_weight);
		poisson.setInputCloud(cloud_pn);

		std::cout << "Reconstructing surface..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		poisson.reconstruct(polygon_mesh);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Surface reconstructed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << "#polygons reconstructed = " <<polygon_mesh.polygons.size() << std::endl;
	}

	// Save the reconstructed surface to a file.
	{
		std::cout << "Saving the reconstructed surface to " << vtk_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		if (pcl::io::saveVTKFile(vtk_filepath, polygon_mesh, /*precision =*/ 5) < 0)
		{
			std::cerr << "The reconstructed surface not saved." << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "The reconstructed surface saved: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}

#if 1
	// Visualize
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	viewer.addPolygonMesh(polygon_mesh, "polygon mesh");
#if 1
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler_cloud(cloud, 255, 0, 0);
	//viewer.addPointCloud<pcl::PointXYZ>(cloud, color_handler_cloud, "point cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> color_handler_cloud(cloud_pn, 255, 0, 0);
	viewer.addPointCloud<pcl::PointNormal>(cloud_pn, color_handler_cloud, "point cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "point cloud");
#endif
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
#endif
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/tools/mls_smoothing.cpp
void mls_smoothing_example()
{
	const std::string input_filepath("./input.pcd");
	const std::string output_filepath("./mls_output.pcd");

	// Load a point cloud from a file
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
		const double search_radius = 0.01;  // Sphere radius to be used for finding the k-nearest neighbors used for fitting
		const double sqr_gauss_param = search_radius * search_radius;  // Parameter used for the distance based weighting of neighbors (recommended = search_radius^2)
		const bool sqr_gauss_param_set = true;
		const int polynomial_order = 2;  // Order of the polynomial to be fit (polynomial_order > 1, indicates using a polynomial fit)

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_pre(new pcl::PointCloud<pcl::PointXYZ>()), cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::fromPCLPointCloud2(*cloud, *cloud_xyz_pre);

		// Filter the NaNs from the cloud
		for (std::size_t i = 0; i < cloud_xyz_pre->size(); ++i)
			if (std::isfinite((*cloud_xyz_pre)[i].x))
				cloud_xyz->push_back((*cloud_xyz_pre)[i]);
		cloud_xyz->header = cloud_xyz_pre->header;
		cloud_xyz->height = 1;
		cloud_xyz->width = cloud_xyz->size();
		cloud_xyz->is_dense = false;

		pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
		mls.setInputCloud(cloud_xyz);
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

		std::cout << "Computing smoothed surface and normals with search_radius = " << mls.getSearchRadius() << ", sqr_gaussian_param = " << mls.getSqrGaussParam() << ", polynomial order = " << mls.getPolynomialOrder() << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn_smoothed(new pcl::PointCloud<pcl::PointNormal>());
		mls.process(*cloud_pn_smoothed);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Smoothed surface and normals computed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud_pn_smoothed->width * cloud_pn_smoothed->height << " points smoothed." << std::endl;

		pcl::toPCLPointCloud2(*cloud_pn_smoothed, cloud_smoothed);
	}

	// Save the smoothed point cloud to a file
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

	// Visualize
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	pcl::visualization::PointCloudGeometryHandlerCustom<pcl::PCLPointCloud2> geometry_handler(pcl::PCLPointCloud2::Ptr(&cloud_smoothed), "x", "y", "z");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PCLPointCloud2> color_handler(pcl::PCLPointCloud2::Ptr(&cloud_smoothed), 150, 150, 150);
	viewer.addPointCloud(
		pcl::PCLPointCloud2::Ptr(&cloud_smoothed),
		pcl::visualization::PointCloudGeometryHandler<pcl::PCLPointCloud2>::Ptr(&geometry_handler), pcl::visualization::PointCloudColorHandler<pcl::PCLPointCloud2>::Ptr(&color_handler),
		Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(),
		"point cloud smoothed"
	);
	//viewer.addPointCloud<pcl::PointXYZ>(cloud, "point cloud");
#if 0
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler_cloud(cloud, 255, 0, 0);
	viewer.addPointCloud<pcl::PointXYZ>(cloud, color_handler_cloud, "point cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "point cloud");
#endif
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
#endif
}

// REF [function] >> mls_smoothing_example()
void mls_smoothing_test()
{
	const std::string input_filepath("./input.pcd");
	const std::string output_filepath("./mls_output.pcd");

	// Load a point cloud from a file
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
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
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn_smoothed(new pcl::PointCloud<pcl::PointNormal>());
	{
		const double search_radius = 0.01;  // Sphere radius to be used for finding the k-nearest neighbors used for fitting
		const double sqr_gauss_param = search_radius * search_radius;  // Parameter used for the distance based weighting of neighbors (recommended = search_radius^2)
		const bool sqr_gauss_param_set = true;
		const int polynomial_order = 2;  // Order of the polynomial to be fit (polynomial_order > 1, indicates using a polynomial fit)

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());

		// Filter the NaNs from the cloud
		for (std::size_t i = 0; i < cloud->size(); ++i)
			if (std::isfinite((*cloud)[i].x))
				cloud_xyz->push_back((*cloud)[i]);
		cloud_xyz->header = cloud->header;
		cloud_xyz->height = 1;
		cloud_xyz->width = cloud_xyz->size();
		cloud_xyz->is_dense = false;

		pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
		mls.setInputCloud(cloud_xyz);
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

		std::cout << "Computing smoothed surface and normals with search_radius = " << mls.getSearchRadius() << ", sqr_gaussian_param = " << mls.getSqrGaussParam() << ", polynomial order = " << mls.getPolynomialOrder() << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		mls.process(*cloud_pn_smoothed);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Smoothed surface and normals computed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud_pn_smoothed->width * cloud_pn_smoothed->height << " points smoothed." << std::endl;
	}

	// Save the smoothed point cloud to a file
	{
		std::cout << "Saving the smoothed point cloud to " << output_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//if (pcl::io::savePCDFile(output_filepath, cloud_pn_smoothed, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), true) < 0)
		pcl::PCDWriter writer;
		if (writer.writeASCII(output_filepath, *cloud_pn_smoothed) != 0)
		{
			std::cerr << "The smoothed point cloud not saved." << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "The smoothed point cloud saved: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}

#if 1
	// Visualize
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	viewer.addPointCloud<pcl::PointNormal>(cloud_pn_smoothed, "point cloud smoothed");
#if 0
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler_cloud(cloud, 255, 0, 0);
	viewer.addPointCloud<pcl::PointXYZ>(cloud, color_handler_cloud, "point cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "point cloud");
#endif
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
	//local::marching_cubes_reconstruction_test();

	// (screened) Poisson surface reconstruction (PSR or SPSR)
	//local::poisson_reconstruction_example();
	local::poisson_reconstruction_test();

	// Moving least squares (MLS)
	//	An algorithm for data smoothing and improved normal estimation
	//local::mls_smoothing_example();
	//local::mls_smoothing_test();
}

}  // namespace my_pcl
