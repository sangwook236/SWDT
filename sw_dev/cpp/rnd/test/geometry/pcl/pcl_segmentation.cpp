#include <ctime>
#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/don.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>
#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/containers/device_array.hpp>
#include <pcl/gpu/segmentation/gpu_extract_clusters.h>
#include <pcl/gpu/segmentation/impl/gpu_extract_clusters.hpp>
#include <vtkPolyLine.h>


namespace {
namespace local {

// REF [site] >> https://pcl.readthedocs.io/en/latest/cluster_extraction.html
void euclidean_cluster_extraction_tutorial()
{
	const auto input_filepath("../table_scene_lms400.pcd");

	// Read in the cloud data.
	pcl::PCDReader reader;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >()), cloud_f(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >());
	reader.read(input_filepath, *cloud);
	std::cout << "PointCloud before filtering has: " << cloud->size() << " data points." << std::endl;

	// Create the filtering object: downsample the dataset using a leaf size of 1cm.
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >());
	vg.setInputCloud(cloud);
	vg.setLeafSize(0.01f, 0.01f, 0.01f);
	vg.filter(*cloud_filtered);
	std::cout << "PointCloud after filtering has: " << cloud_filtered->size() << " data points." << std::endl;

	// Create the segmentation object for the planar model and set all the parameters.
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	pcl::PointIndices::Ptr inliers(std::make_shared<pcl::PointIndices>());
	pcl::ModelCoefficients::Ptr coefficients(std::make_shared<pcl::ModelCoefficients>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >());
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(100);
	seg.setDistanceThreshold(0.02);

	int nr_points = (int)cloud_filtered->size();
	while (double(cloud_filtered->size()) > 0.3 * nr_points)
	{
		// Segment the largest planar component from the remaining cloud.
		seg.setInputCloud(cloud_filtered);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() == 0)
		{
			std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
			break;
		}

		// Extract the planar inliers from the input cloud.
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(cloud_filtered);
		extract.setIndices(inliers);
		extract.setNegative(false);

		// Get the points associated with the planar surface.
		extract.filter(*cloud_plane);
		std::cout << "PointCloud representing the planar component: " << cloud_plane->size() << " data points." << std::endl;

		// Remove the planar inliers, extract the rest.
		extract.setNegative(true);
		extract.filter(*cloud_f);
		*cloud_filtered = *cloud_f;
	}

	// Creating the KdTree object for the search method of the extraction.
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(std::make_shared<pcl::search::KdTree<pcl::PointXYZ> >());
	tree->setInputCloud(cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(0.02);  // Search radius. 2cm.
	ec.setMinClusterSize(100);
	ec.setMaxClusterSize(25000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_filtered);
	ec.extract(cluster_indices);

	pcl::PCDWriter writer;
	int j = 0;
	for (const auto& cluster : cluster_indices)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >());
		for (const auto& idx : cluster.indices)
		{
			cloud_cluster->push_back((*cloud_filtered)[idx]);
		}
		cloud_cluster->width = cloud_cluster->size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
		std::stringstream ss;
		ss << std::setw(4) << std::setfill('0') << j;
		writer.write<pcl::PointXYZ>("./cloud_cluster_" + ss.str() + ".pcd", *cloud_cluster, false);
		++j;
	}
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/gpu/examples/segmentation/src/seg.cpp
void euclidean_cluster_extraction_gpu_example()
{
	const std::string pcd_filepath("./input.pcd");

	pcl::PCDWriter writer;

	// Read in the cloud data
	pcl::PCDReader reader;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	reader.read(pcd_filepath, *cloud_filtered);

	pcl::Indices unused;
	pcl::removeNaNFromPointCloud(*cloud_filtered, *cloud_filtered, unused);

	//-----
	// CPU version

	std::cout << "INFO: PointCloud_filtered still has " << cloud_filtered->size() << " Points " << std::endl;
	clock_t tStart = clock();

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(0.02);  // Search radius. 2cm
	ec.setMinClusterSize(100);
	ec.setMaxClusterSize(25000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_filtered);
	ec.extract(cluster_indices);
	
	printf("CPU Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	int j = 0;
	for (const pcl::PointIndices &cluster: cluster_indices)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (const auto &index: (cluster.indices))
			cloud_cluster->push_back((*cloud_filtered)[index]);  //*
		cloud_cluster->width = cloud_cluster->size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
		std::stringstream ss;
		ss << "cloud_cluster_" << j << ".pcd";
		writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster, false);  //*
		++j;
	}

	//-----
	// GPU version

	std::cout << "INFO: starting with the GPU version" << std::endl;
	tStart = clock();

	pcl::gpu::Octree::PointCloud cloud_device;
	cloud_device.upload(cloud_filtered->points);
	
	pcl::gpu::Octree::Ptr octree_device(new pcl::gpu::Octree);
	octree_device->setCloud(cloud_device);
	octree_device->build();

	std::vector<pcl::PointIndices> cluster_indices_gpu;
	pcl::gpu::EuclideanClusterExtraction<pcl::PointXYZ> gec;
	//pcl::gpu::EuclideanLabeledClusterExtraction<pcl::PointXYZ> gec;
	gec.setClusterTolerance(0.02);  // Search radius. 2cm
	gec.setMinClusterSize(100);
	gec.setMaxClusterSize(25000);
	gec.setSearchMethod(octree_device);  // NOTE [info] >> not a k-d tree
	gec.setHostCloud(cloud_filtered);  // NOTE [info] >> not a cloud on the device, but a cloud on the host
	gec.extract(cluster_indices_gpu);
	//octree_device.clear();

	printf("GPU Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	std::cout << "INFO: stopped with the GPU version" << std::endl;

	j = 0;
	for (const pcl::PointIndices &cluster: cluster_indices_gpu)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_gpu(new pcl::PointCloud<pcl::PointXYZ>);
		for (const auto &index: (cluster.indices))
			cloud_cluster_gpu->push_back((*cloud_filtered)[index]);  //*
		cloud_cluster_gpu->width = cloud_cluster_gpu->size();
		cloud_cluster_gpu->height = 1;
		cloud_cluster_gpu->is_dense = true;

		std::cout << "PointCloud representing the Cluster: " << cloud_cluster_gpu->size() << " data points." << std::endl;
		std::stringstream ss;
		ss << "gpu_cloud_cluster_" << j << ".pcd";
		writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster_gpu, false);  //*
		++j;
	}
}

// REF [site] >> https://pcl.readthedocs.io/en/latest/region_growing_segmentation.html
void region_growing_segmentation_tutorial()
{
	const auto input_filepath("../region_growing_tutorial.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (std::make_shared<pcl::PointCloud<pcl::PointXYZ> >());
	if ( pcl::io::loadPCDFile<pcl::PointXYZ>(input_filepath, *cloud) == -1)
	{
		std::cout << "Failed to load a point cloud, " << input_filepath << std::endl;
		return;
	}

	pcl::search::Search<pcl::PointXYZ>::Ptr tree(std::make_shared<pcl::search::KdTree<pcl::PointXYZ> >());
	pcl::PointCloud<pcl::Normal>::Ptr normals(std::make_shared<pcl::PointCloud<pcl::Normal> >());
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
	normal_estimator.setSearchMethod(tree);
	normal_estimator.setInputCloud(cloud);
	normal_estimator.setKSearch(50);
	normal_estimator.compute(*normals);

	pcl::IndicesPtr indices(std::make_shared<std::vector<int> >());
	pcl::removeNaNFromPointCloud(*cloud, *indices);

	pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
	reg.setMinClusterSize(50);
	reg.setMaxClusterSize(1000000);
	reg.setSearchMethod(tree);
	reg.setNumberOfNeighbours(30);
	reg.setInputCloud(cloud);
	reg.setIndices(indices);
	reg.setInputNormals(normals);
	reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
	reg.setCurvatureThreshold(1.0);

	std::vector<pcl::PointIndices> clusters;
	reg.extract(clusters);

	std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;
	std::cout << "First cluster has " << clusters[0].indices.size() << " points." << std::endl;
	std::cout << "These are the indices of the points of the initial" << std::endl << "cloud that belong to the first cluster:" << std::endl;
	std::size_t counter = 0;
	while (counter < clusters[0].indices.size())
	{
		std::cout << clusters[0].indices[counter] << ", ";
		++counter;
		if (counter % 10 == 0)
			std::cout << std::endl;
	}
	std::cout << std::endl;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
	pcl::visualization::CloudViewer viewer("Cluster viewer");
	viewer.showCloud(colored_cloud);
	while (!viewer.wasStopped())
	{
		//viewer->spinOnce(100);
	}
}

// REF [site] >> https://pcl.readthedocs.io/en/latest/min_cut_segmentation.html
void min_cut_segmentation_tutorial()
{
	const auto input_filepath("../min_cut_segmentation_tutorial.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >());
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_filepath, *cloud) == -1)
	{
		std::cout << "Failed to load a cloud file, " << input_filepath << std::endl;
		return;
	}

	pcl::IndicesPtr indices(std::make_shared<std::vector<int> >());
	pcl::removeNaNFromPointCloud(*cloud, *indices);

	pcl::MinCutSegmentation<pcl::PointXYZ> seg;
	seg.setInputCloud(cloud);
	seg.setIndices(indices);

	pcl::PointCloud<pcl::PointXYZ>::Ptr foreground_points(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >());
	pcl::PointXYZ point;
	point.x = 68.97;
	point.y = -18.55;
	point.z = 0.57;
	foreground_points->points.push_back(point);
	seg.setForegroundPoints(foreground_points);

	seg.setSigma(0.25);
	seg.setRadius(3.0433856);
	seg.setNumberOfNeighbours(14);
	seg.setSourceWeight(0.8);

	std::vector<pcl::PointIndices> clusters;
	seg.extract(clusters);

	std::cout << "Maximum flow is " << seg.getMaxFlow() << std::endl;

	pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = seg.getColoredCloud();
	pcl::visualization::CloudViewer viewer("Cluster viewer");
	viewer.showCloud(colored_cloud);
	while (!viewer.wasStopped())
	{
		//viewer->spinOnce(100);
	}
}

bool enforceIntensitySimilarity(const pcl::PointXYZINormal& point_a, const pcl::PointXYZINormal& point_b, float /*squared_distance*/)
{
	if (std::abs(point_a.intensity - point_b.intensity) < 5.0f)
		return (true);
	else
		return (false);
}

bool enforceNormalOrIntensitySimilarity(const pcl::PointXYZINormal& point_a, const pcl::PointXYZINormal& point_b, float /*squared_distance*/)
{
	Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap(), point_b_normal = point_b.getNormalVector3fMap();
	if (std::abs(point_a.intensity - point_b.intensity) < 5.0f)
		return (true);
	if (std::abs(point_a_normal.dot(point_b_normal)) > std::cos(30.0f / 180.0f * static_cast<float>(M_PI)))
		return (true);
	return (false);
}

bool customRegionGrowing(const pcl::PointXYZINormal& point_a, const pcl::PointXYZINormal& point_b, float squared_distance)
{
	Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap(), point_b_normal = point_b.getNormalVector3fMap();
	if (squared_distance < 10000)
	{
		if (std::abs(point_a.intensity - point_b.intensity) < 8.0f)
			return (true);
		if (std::abs(point_a_normal.dot(point_b_normal)) > std::cos(30.0f / 180.0f * static_cast<float>(M_PI)))
			return (true);
	}
	else
	{
		if (std::abs(point_a.intensity - point_b.intensity) < 3.0f)
			return (true);
	}
	return (false);
}

// REF [site] >> https://pcl.readthedocs.io/en/latest/conditional_euclidean_clustering.html
void conditional_euclidean_clustering_tutorial()
{
	const auto input_filepath("../Statues_4.pcd");

	// Data containers used.
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(std::make_shared<pcl::PointCloud<pcl::PointXYZI> >()), cloud_out(std::make_shared<pcl::PointCloud<pcl::PointXYZI> >());
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(std::make_shared<pcl::PointCloud<pcl::PointXYZINormal> >());
	pcl::IndicesClustersPtr clusters(std::make_shared<pcl::IndicesClusters>()), small_clusters(std::make_shared<pcl::IndicesClusters>()), large_clusters(std::make_shared<pcl::IndicesClusters>());
	pcl::search::KdTree<pcl::PointXYZI>::Ptr search_tree(std::make_shared<pcl::search::KdTree<pcl::PointXYZI> >());
	pcl::console::TicToc tt;

	// Load the input point cloud.
	std::cerr << "Loading...\n", tt.tic();
	pcl::io::loadPCDFile(input_filepath, *cloud_in);
	std::cerr << ">> Done: " << tt.toc() << " ms, " << cloud_in->size() << " points\n";

	// Downsample the cloud using a Voxel Grid class.
	std::cerr << "Downsampling...\n", tt.tic();
	pcl::VoxelGrid<pcl::PointXYZI> vg;
	vg.setInputCloud(cloud_in);
	vg.setLeafSize(80.0, 80.0, 80.0);
	vg.setDownsampleAllData(true);
	vg.filter(*cloud_out);
	std::cerr << ">> Done: " << tt.toc() << " ms, " << cloud_out->size() << " points\n";

	// Set up a Normal Estimation class and merge data in cloud_with_normals.
	std::cerr << "Computing normals...\n", tt.tic();
	pcl::copyPointCloud(*cloud_out, *cloud_with_normals);
	pcl::NormalEstimation<pcl::PointXYZI, pcl::PointXYZINormal> ne;
	ne.setInputCloud(cloud_out);
	ne.setSearchMethod(search_tree);
	ne.setRadiusSearch(300.0);
	ne.compute(*cloud_with_normals);
	std::cerr << ">> Done: " << tt.toc() << " ms\n";

	// Set up a Conditional Euclidean Clustering class.
	std::cerr << "Segmenting to clusters...\n", tt.tic();
	pcl::ConditionalEuclideanClustering<pcl::PointXYZINormal> cec(true);
	cec.setInputCloud(cloud_with_normals);
	cec.setConditionFunction(&customRegionGrowing);
	cec.setClusterTolerance(500.0);
	cec.setMinClusterSize(cloud_with_normals->size() / 1000);
	cec.setMaxClusterSize(cloud_with_normals->size() / 5);
	cec.segment(*clusters);
	cec.getRemovedClusters(small_clusters, large_clusters);
	std::cerr << ">> Done: " << tt.toc() << " ms\n";

	// Using the intensity channel for lazy visualization of the output.
	for (const auto& small_cluster : (*small_clusters))
		for (const auto& j : small_cluster.indices)
			(*cloud_out)[j].intensity = -2.0;
	for (const auto& large_cluster : (*large_clusters))
		for (const auto& j : large_cluster.indices)
			(*cloud_out)[j].intensity = +10.0;
	for (const auto& cluster : (*clusters))
	{
		int label = rand () % 8;
		for (const auto& j : cluster.indices)
			(*cloud_out)[j].intensity = label;
	}

	// Save the output point cloud.
	std::cerr << "Saving...\n", tt.tic();
	pcl::io::savePCDFile("./output.pcd", *cloud_out);
	std::cerr << ">> Done: " << tt.toc() << " ms\n";
}

// REF [site] >> https://pcl.readthedocs.io/en/latest/don_segmentation.html
void don_segmentation_tutorial()
{
	//const auto input_filepath("/path/to/sample.pcd");
	const auto input_filepath("../../20230621/crown.pcd");
	//const auto input_filepath("../../20230621/inlay.pcd");
	//const auto input_filepath("../../20230621/onlay.pcd");

	// The smallest scale to use in the DoN filter.
	const double scale1 = 0.1;
	// The largest scale to use in the DoN filter.
	const double scale2 = 0.2;
	// The minimum DoN magnitude to threshold by.
	const double threshold = 0.1;
	// Segment scene into clusters with given distance tolerance using euclidean clustering.
	const double segradius = 1.0;

	// Load cloud in blob format.
	pcl::PCLPointCloud2 blob;
	pcl::io::loadPCDFile(input_filepath, blob);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(std::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >());
	pcl::fromPCLPointCloud2(blob, *cloud);

	// Create a search tree, use KDTreee for non-organized data.
	pcl::search::Search<pcl::PointXYZRGB>::Ptr tree;
	if (cloud->isOrganized())
	{
		tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZRGB>());
	}
	else
	{
		tree.reset(new pcl::search::KdTree<pcl::PointXYZRGB>(false));
	}

	// Set the input pointcloud for the search tree.
	tree->setInputCloud(cloud);

	if (scale1 >= scale2)
	{
		std::cerr << "Error: Large scale must be > small scale!" << std::endl;
		return;
	}

	// Compute normals using both small and large scales at each point.
	pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::PointNormal> ne;
	ne.setInputCloud(cloud);
	ne.setSearchMethod(tree);

	// Setting viewpoint is very important, so that we can ensure normals are all pointed in the same direction!
	ne.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

	// Calculate normals with the small scale.
	std::cout << "Calculating normals for scale..." << scale1 << std::endl;
	pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale(std::make_shared<pcl::PointCloud<pcl::PointNormal> >());

	ne.setRadiusSearch(scale1);
	ne.compute(*normals_small_scale);

	// Calculate normals with the large scale.
	std::cout << "Calculating normals for scale..." << scale2 << std::endl;
	pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale(std::make_shared<pcl::PointCloud<pcl::PointNormal> >());

	ne.setRadiusSearch(scale2);
	ne.compute(*normals_large_scale);

	// Create output cloud for DoN results.
	pcl::PointCloud<pcl::PointNormal>::Ptr doncloud(std::make_shared<pcl::PointCloud<pcl::PointNormal> >());
	copyPointCloud(*cloud, *doncloud);

	std::cout << "Calculating DoN..." << std::endl;
	// Create DoN operator.
	pcl::DifferenceOfNormalsEstimation<pcl::PointXYZRGB, pcl::PointNormal, pcl::PointNormal> don;
	don.setInputCloud(cloud);
	don.setNormalScaleLarge(normals_large_scale);
	don.setNormalScaleSmall(normals_small_scale);

	if (!don.initCompute())
	{
		std::cerr << "Error: Could not initialize DoN feature operator." << std::endl;
		return;
	}

	// Compute DoN.
	don.computeFeature(*doncloud);

	// Save DoN features.
	pcl::PCDWriter writer;
	writer.write<pcl::PointNormal>("./don.pcd", *doncloud, false); 

	// Filter by magnitude.
	std::cout << "Filtering out DoN mag <= " << threshold << "..." << std::endl;

	// Build the condition for filtering.
	pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond(std::make_shared<pcl::ConditionOr<pcl::PointNormal> >());
	range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
		std::make_shared<pcl::FieldComparison<pcl::PointNormal> >("curvature", pcl::ComparisonOps::GT, threshold)
	));
	// Build the filter.
	pcl::ConditionalRemoval<pcl::PointNormal> condrem;
	condrem.setCondition(range_cond);
	condrem.setInputCloud(doncloud);

	pcl::PointCloud<pcl::PointNormal>::Ptr doncloud_filtered(std::make_shared<pcl::PointCloud<pcl::PointNormal> >());

	// Apply filter.
	condrem.filter(*doncloud_filtered);

	doncloud = doncloud_filtered;

	// Save filtered output.
	std::cout << "Filtered Pointcloud: " << doncloud->size() << " data points." << std::endl;

	writer.write<pcl::PointNormal>("./don_filtered.pcd", *doncloud, false); 

	// Filter by magnitude.
	std::cout << "Clustering using EuclideanClusterExtraction with tolerance <= " << segradius << "..." << std::endl;

	pcl::search::KdTree<pcl::PointNormal>::Ptr segtree(std::make_shared<pcl::search::KdTree<pcl::PointNormal> >());
	segtree->setInputCloud(doncloud);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointNormal> ec;

	ec.setClusterTolerance(segradius);
	ec.setMinClusterSize(50);
	ec.setMaxClusterSize(100000);
	ec.setSearchMethod(segtree);
	ec.setInputCloud(doncloud);
	ec.extract(cluster_indices);

	int j = 0;
	for (const auto& cluster : cluster_indices)
	{
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_cluster_don(std::make_shared<pcl::PointCloud<pcl::PointNormal> >());
		for (const auto& idx : cluster.indices)
		{
			cloud_cluster_don->points.push_back((*doncloud)[idx]);
		}

		cloud_cluster_don->width = cloud_cluster_don->size();
		cloud_cluster_don->height = 1;
		cloud_cluster_don->is_dense = true;

		// Save cluster.
		std::cout << "PointCloud representing the Cluster: " << cloud_cluster_don->size() << " data points." << std::endl;
		std::stringstream ss;
		ss << "./don_cluster_" << j << ".pcd";
		writer.write<pcl::PointNormal>(ss.str(), *cloud_cluster_don, false);
		++j;
	}
}

void addSupervoxelConnectionsToViewer(
	pcl::PointXYZRGBA &supervoxel_center,
	pcl::PointCloud<pcl::PointXYZRGBA> &adjacent_supervoxel_centers,
	const std::string &supervoxel_name,
	pcl::visualization::PCLVisualizer::Ptr &viewer
)
{
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();

	// Iterate through all adjacent points, and add a center point to adjacent point pair.
	for (auto adjacent_itr = adjacent_supervoxel_centers.begin(); adjacent_itr != adjacent_supervoxel_centers.end(); ++adjacent_itr)
	{
		points->InsertNextPoint(supervoxel_center.data);
		points->InsertNextPoint(adjacent_itr->data);
	}

	// Create a polydata to store everything in.
	vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
	// Add the points to the dataset.
	polyData->SetPoints(points);
	polyLine->GetPointIds()->SetNumberOfIds(points->GetNumberOfPoints());
	for(unsigned int i = 0; i < points->GetNumberOfPoints(); ++i)
		polyLine->GetPointIds()->SetId(i, i);
	cells->InsertNextCell(polyLine);

	// Add the lines to the dataset.
	polyData->SetLines(cells);
	viewer->addModelFromPolyData(polyData, supervoxel_name);
}

// REF [site] >> https://pcl.readthedocs.io/en/latest/supervoxel_clustering.html
void supervoxels_clustering_tutorial()
{
	const std::string input_filepath("../milk_cartoon_all_small_clorox.pcd");

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(std::make_shared<pcl::PointCloud<pcl::PointXYZRGBA> >());
	std::cout << "Loading point cloud..." << std::endl;
	if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(input_filepath, *cloud))
	{
		std::cerr << "Failed to load a point cloud file, " << input_filepath << std::endl;
		return;
	}

	const bool disable_transform = true;
	const float voxel_resolution = 0.008f;
	const float seed_resolution = 0.1f;
	const float color_importance = 0.2f;
	const float spatial_importance = 0.4f;
	const float normal_importance = 1.0f;

	// How to use supervoxels.

	pcl::SupervoxelClustering<pcl::PointXYZRGBA> super(voxel_resolution, seed_resolution);
	if (disable_transform)
		super.setUseSingleCameraTransform(false);
	super.setInputCloud(cloud);
	super.setColorImportance(color_importance);
	super.setSpatialImportance(spatial_importance);
	super.setNormalImportance(normal_importance);

	std::map<std::uint32_t, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr> supervoxel_clusters;

	std::cout << "Extracting supervoxels!" << std::endl;
	super.extract(supervoxel_clusters);
	std::cout << "Found " << supervoxel_clusters.size() << " supervoxels." << std::endl;

	pcl::visualization::PCLVisualizer::Ptr viewer(std::make_shared<pcl::visualization::PCLVisualizer>("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr voxel_centroid_cloud = super.getVoxelCentroidCloud();
	viewer->addPointCloud(voxel_centroid_cloud, "voxel centroids");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "voxel centroids");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.95, "voxel centroids");

	pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud();
	viewer->addPointCloud(labeled_voxel_cloud, "labeled voxels");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.8, "labeled voxels");

	pcl::PointCloud<pcl::PointNormal>::Ptr sv_normal_cloud = super.makeSupervoxelNormalCloud(supervoxel_clusters);
	// We have this disabled so graph is easy to see, uncomment to see supervoxel normals.
	//viewer->addPointCloudNormals<PointNormal>(sv_normal_cloud, 1, 0.05f, "supervoxel_normals");

	std::cout << "Getting supervoxel adjacency..." << std::endl;
	std::multimap<std::uint32_t, std::uint32_t> supervoxel_adjacency;
	super.getSupervoxelAdjacency(supervoxel_adjacency);
	// To make a graph of the supervoxel adjacency, we need to iterate through the supervoxel adjacency multimap.
	for (auto label_itr = supervoxel_adjacency.cbegin(); label_itr != supervoxel_adjacency.cend(); )
	{
		// First get the label.
		std::uint32_t supervoxel_label = label_itr->first;
		// Now get the supervoxel corresponding to the label.
		pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr supervoxel = supervoxel_clusters.at(supervoxel_label);

		// Now we need to iterate through the adjacent supervoxels and make a point cloud of them.
		pcl::PointCloud<pcl::PointXYZRGBA> adjacent_supervoxel_centers;
		for (auto adjacent_itr = supervoxel_adjacency.equal_range(supervoxel_label).first; adjacent_itr!=supervoxel_adjacency.equal_range(supervoxel_label).second; ++adjacent_itr)
		{
			pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr neighbor_supervoxel = supervoxel_clusters.at(adjacent_itr->second);
			adjacent_supervoxel_centers.push_back(neighbor_supervoxel->centroid_);
		}
		// Now we make a name for this polygon.
		std::stringstream ss;
		ss << "supervoxel_" << supervoxel_label;
		// This function is shown below, but is beyond the scope of this tutorial - basically it just generates a "star" polygon mesh from the points given.
		addSupervoxelConnectionsToViewer(supervoxel->centroid_, adjacent_supervoxel_centers, ss.str(), viewer);
		// Move iterator forward to next label.
		label_itr = supervoxel_adjacency.upper_bound(supervoxel_label);
	}

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/examples/segmentation/example_region_growing.cpp
void region_growing_example()
{
	//const auto input_filepath("./region_growing_tutorial.pcd");
	const auto input_filepath("../../20230621/crown.pcd");
	//const auto input_filepath("../../20230621/inlay.pcd");
	//const auto input_filepath("../../20230621/onlay.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_nans(std::make_shared<pcl::PointCloud<pcl::PointXYZ> >());
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(std::make_shared<pcl::PointCloud<pcl::Normal> >());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_segmented(std::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >());

	if (pcl::io::loadPCDFile(input_filepath, *cloud_ptr) == -1)
	{
		std::cerr << "Failed to load a point cloud file, " << input_filepath << std::endl;
		return;
	}
	std::cout << "Loaded cloud " << input_filepath << " of size " << static_cast<std::size_t>(cloud_ptr->size()) << std::endl;

	// Remove the NaNs.
	cloud_ptr->is_dense = false;
	cloud_no_nans->is_dense = false;
	pcl::Indices indices;
	pcl::removeNaNFromPointCloud(*cloud_ptr, *cloud_no_nans, indices);
	std::cout << "Removed NaNs from " << static_cast<std::size_t>(cloud_ptr->size()) << " to " << static_cast<std::size_t>(cloud_no_nans->size()) << std::endl;

	// Estimate the normals.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud_no_nans);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(std::make_shared<pcl::search::KdTree<pcl::PointXYZ> >());
	ne.setSearchMethod(tree_n);
	ne.setRadiusSearch(1.0);
	ne.compute(*cloud_normals);
	std::cout << "Normals are computed and size is " << static_cast<std::size_t>(cloud_normals->size()) << std::endl;

	// Region growing.
	pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> rg;
	rg.setSmoothModeFlag(false);  // Depends on the cloud being processed.
	rg.setInputCloud(cloud_no_nans);
	rg.setInputNormals(cloud_normals);

	const auto start_time(std::chrono::high_resolution_clock::now());
	std::vector<pcl::PointIndices> clusters;
	rg.extract(clusters);
	const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
	std::cout << "Region growing done: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	cloud_segmented = rg.getColoredCloud();
	std::cout << "Number of segments done = " << static_cast<std::size_t>(clusters.size()) << std::endl;

	// Writing the resulting cloud into a pcd file.
	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZRGB>("./segment_result.pcd", *cloud_segmented, false);
	std::cout << "Segmentation results written to ./segment_result.pcd." << std::endl;

	if (false)
	{
		std::cout << "Writing clusters to clusters.dat..." << std::endl;
		std::ofstream clusters_file;
		clusters_file.open("./clusters.dat");
		for (std::size_t i = 0; i < clusters.size(); ++i)
		{
			clusters_file << i << "#" << clusters[i].indices.size() << ":";
			for (const auto& cluster_idx : clusters[i].indices)
				clusters_file << " " << cluster_idx;
			clusters_file << std::endl;
		}
		clusters_file.close();
	}
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/examples/segmentation/example_supervoxels.cpp
void supervoxels_example()
{
	throw std::runtime_error("Not yet implemented");
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/examples/segmentation/example_lccp_segmentation.cpp
void lccp_segmentatio_example()
{
	throw std::runtime_error("Not yet implemented");
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/examples/segmentation/example_cpc_segmentation.cpp
void cpc_segmentatio_example()
{
	throw std::runtime_error("Not yet implemented");
}

void grabcut_test()
{
	throw std::runtime_error("Not yet implemented");
}

// REF [site] >>
//	https://en.wikipedia.org/wiki/DBSCAN
//	https://github.com/Eleobert/dbscan/blob/master/dbscan.cpp
auto dbscan(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const double radius, const size_t min_pts, const double octree_resolution)
{
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(octree_resolution);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();

	const auto &num_points = cloud->size();
	std::vector<std::vector<size_t>> clusters;
	std::vector<char> visited(num_points, 0);  // 0: undefined, 1: noisy, 2: clustered
	pcl::Indices k_indices, k_sub_indices;
	std::vector<float> k_sqr_distances;

	for (size_t pt_idx = 0; pt_idx < num_points; ++pt_idx)
//#pragma omp parallel for private(k_indices, k_sub_indices, k_sqr_distances) shared(octree, cloud, visited)  // Not correctly working
//	for (int pt_idx = 0; pt_idx < num_points; ++pt_idx)
	{
		if (visited[pt_idx]) continue;

		octree.radiusSearch(cloud->points[pt_idx], radius, k_indices, k_sqr_distances);
		//octree.nearestKSearch(cloud->points[pt_idx], k, k_indices, k_sqr_distances);
		//octree.approxNearestSearch(cloud->points[pt_idx], k_index, k_sqr_distance);

		if (k_indices.size() < min_pts)  // Noise
		{
			visited[pt_idx] = 1;
			continue;
		}
		visited[pt_idx] = 2;

		std::vector<size_t> cluster({pt_idx});
//		std::vector<size_t> cluster({size_t(pt_idx)});
		while (!k_indices.empty())
		{
			const auto neighbor_idx(k_indices.back());
			k_indices.pop_back();
			if (1 == visited[neighbor_idx])  // If noise
			{
				visited[neighbor_idx] = 2;
				cluster.push_back(neighbor_idx);
				continue;
			}
			if (visited[neighbor_idx]) continue;
			visited[neighbor_idx] = 2;
			cluster.push_back(neighbor_idx);

			octree.radiusSearch(cloud->points[neighbor_idx], radius, k_sub_indices, k_sqr_distances);
			//octree.nearestKSearch(cloud->points[neighbor_idx], k, k_sub_indices, k_sqr_distances);
			//octree.approxNearestSearch(cloud->points[neighbor_idx], k_index, k_sqr_distance);

#if 0
			// For checking
			if (k_sub_indices.size() >= min_pts)
			{
				for (auto &clust: clusters)
					std::sort(clust.begin(), clust.end());
				for (const auto &idx: k_sub_indices)
					if (2 == visited[idx])
					{
						size_t clust_idx = 0;
						for (const auto &clust: clusters)
						{
							if (std::find(clust.begin(), clust.end(), idx) != clust.end())
							{
								pcl::Indices k_indices_tmp;
								octree.radiusSearch(cloud->points[idx], radius, k_indices_tmp, k_sqr_distances);
								//octree.nearestKSearch(cloud->points[idx], k, k_indices_tmp, k_sqr_distances);
								//octree.approxNearestSearch(cloud->points[idx], k_index_tmp, k_sqr_distance);
								std::cout << "Candidate point #" << idx << " (neighbor of point #" << neighbor_idx << ") of cluster #" << clusters.size() << " is already assigned to cluster #" << clust_idx << ": #neighbors = " << k_indices_tmp.size() << std::endl;
								//break;
							}
							++clust_idx;
						}
					}
			}
#endif

			if (k_sub_indices.size() >= min_pts)
#if 1
				std::copy(k_sub_indices.begin(), k_sub_indices.end(), std::back_inserter(k_indices));
#else
				std::copy_if(k_sub_indices.begin(), k_sub_indices.end(), std::back_inserter(k_indices), [&visited](const auto &idx) {
					//return !visited[idx];
					return 2 != visited[idx];
				});
#endif
		}

//#pragma omp critical
		if (cluster.size() >= min_pts)
			clusters.emplace_back(std::move(cluster));
	}

	// Sort clusters
#if 0
	for (auto &cluster: clusters)
		std::sort(cluster.begin(), cluster.end());
#elif 0
	std::for_each(clusters.begin(), clusters.end(), [](auto &cluster) {
		std::sort(cluster.begin(), cluster.end());
	});
#else
	// No sorting
#endif

	return clusters;
}

// REF [site] >>
//	https://en.wikipedia.org/wiki/DBSCAN
//	https://github.com/Eleobert/dbscan/blob/master/dbscan.cpp
auto dbscan_gpu(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const float radius, const size_t min_pts)
{
	//const int k(1);  // OctreeGPU::knnSearch is supported only for k == 1
	const int max_search_results(std::max<int>(cloud->size() / 1000, 100));
	//std::cout << "#max search results = " << max_search_results << std::endl;

	pcl::gpu::Octree::PointCloud cloud_device;
	cloud_device.upload(cloud->points);

	pcl::gpu::Octree octree_device;
	octree_device.setCloud(cloud_device);
	//std::cout << "Building an octree on GPU..." << std::endl;
	//const auto start_time(std::chrono::high_resolution_clock::now());
	octree_device.build();
	//const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
	//std::cout << "An octree built on GPU: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;

	pcl::gpu::Octree::Queries queries_device;
	pcl::gpu::NeighborIndices result_device;
	//pcl::gpu::Octree::ResultSqrDists sqr_distance_device;

	const auto &num_points = cloud->size();
	std::vector<std::vector<size_t>> clusters;
	std::vector<char> visited(num_points, 0);  // 0: undefined, 1: noisy, 2: clustered
	std::list<int> k_indices;
	std::vector<int> k_indices_sizes, k_indices_data, k_sub_indices_sizes, k_sub_indices_data;
	for (size_t pt_idx = 0; pt_idx < num_points; ++pt_idx)
	{
		if (visited[pt_idx]) continue;

		std::vector<pcl::PointXYZ> queries({cloud->points[pt_idx]});  // TODO [check] >>
		queries_device.upload(queries);  // Slow
		result_device.create(queries_device.size(), max_search_results);  // Slow

		octree_device.radiusSearch(queries_device, radius, max_search_results, result_device);
		//octree_device.nearestKSearchBatch(queries_device, k, result_device);
		//octree_device.approxNearestSearch(queries_device, result_device, sqr_distance_device);

		//std::vector<int> k_indices_sizes, k_indices_data;
		result_device.sizes.download(k_indices_sizes);  // Slow
		result_device.data.download(k_indices_data);  // Slow
		//assert(result_device.sizes.size() == result_device.data.size());
		assert(queries_device.size() == k_indices_sizes.size());
		if (k_indices_sizes[0] < min_pts)  // Noise
		{
			visited[pt_idx] = 1;
			continue;
		}
		visited[pt_idx] = 2;

		auto it_end = k_indices_data.begin();
		std::advance(it_end, k_indices_sizes[0]);
		k_indices.assign(k_indices_data.begin(), it_end);

		std::vector<size_t> cluster({pt_idx});
		while (!k_indices.empty())
		{
			std::vector<pcl::PointXYZ> queries;  // TODO [check] >>
			queries.reserve(k_indices.size());
			//std::vector<int> query_point_indices;  // For checking
			//query_point_indices.reserve(k_indices.size());  // For checking
			for (const auto &neighbor_idx: k_indices)
			{
				if (1 == visited[neighbor_idx])  // If noise
				{
					visited[neighbor_idx] = 2;
					cluster.push_back(neighbor_idx);
					continue;
				}
				if (visited[neighbor_idx]) continue;
				visited[neighbor_idx] = 2;
				cluster.push_back(neighbor_idx);

				queries.push_back(cloud->points[neighbor_idx]);
				//query_point_indices.push_back(neighbor_idx);  // For checking
			}
			k_indices.clear();

			if (!queries.empty())
			{
				queries_device.upload(queries);  // Slow
				result_device.create(queries_device.size(), max_search_results);  // Slow

				octree_device.radiusSearch(queries_device, radius, max_search_results, result_device);
				//octree_device.nearestKSearchBatch(queries_device, k, result_device);
				//octree_device.approxNearestSearch(queries_device, result_device, sqr_distance_device);

				//std::vector<int> k_sub_indices_sizes, k_sub_indices_data;  // Too slow
				result_device.sizes.download(k_sub_indices_sizes);  // Slow
				result_device.data.download(k_sub_indices_data);  // Slow
				//assert(result_device.sizes.size() == result_device.data.size());
				assert(queries_device.size() == k_sub_indices_sizes.size());

#if 0
				// For checking
				for (auto &clust: clusters)
					std::sort(clust.begin(), clust.end());
				for (std::size_t si = 0; si < k_sub_indices_sizes.size(); ++si)
					for (std::size_t sj = 0; sj < k_sub_indices_sizes[si]; ++sj)
					{
						const auto &idx = k_sub_indices_data[si * max_search_results + sj];
						if (2 == visited[idx])
						{
							size_t clust_idx = 0;
							for (const auto &clust: clusters)
							{
								if (std::find(clust.begin(), clust.end(), idx) != clust.end())
								{
									std::vector<pcl::PointXYZ> queries({cloud->points[idx]});  // TODO [check] >>
									queries_device.upload(queries);
									result_device.create(queries_device.size(), max_search_results);
									octree_device.radiusSearch(queries_device, radius, max_search_results, result_device);
									//octree_device.nearestKSearchBatch(queries_device, k, result_device);
									//octree_device.approxNearestSearch(queries_device, result_device, sqr_distance_device);
									std::vector<int> k_indices_sizes_tmp;  // TODO [check] >>
									result_device.sizes.download(k_indices_sizes_tmp);
									std::cout << "Candidate point #" << idx << " (neighbor of point #" << query_point_indices[si] << ") of cluster #" << clusters.size() << " is already assigned to cluster #" << clust_idx << ": #neighbors = " << k_indices_sizes_tmp[0] << std::endl;
									//break;
								}
								++clust_idx;
							}
						}
					}
#endif

				for (std::size_t si = 0; si < k_sub_indices_sizes.size(); ++si)
					if (k_sub_indices_sizes[si] >= min_pts)
#if 1
						//for (std::size_t sj = 0; sj < k_sub_indices_sizes[si]; ++sj)
						//	k_indices.push_back(k_sub_indices_data[si * max_search_results + sj]);
						std::copy(&k_sub_indices_data[si * max_search_results], &k_sub_indices_data[si * max_search_results] + k_sub_indices_sizes[si], std::back_inserter(k_indices));
#else
					{
						auto it_begin = k_indices_data.begin(), it_end = k_indices_data.begin();
						std::advance(it_begin, si * max_search_results);
						std::advance(it_end, si * max_search_results + k_indices_sizes[si]);
#if 1
						std::copy(it_begin, it_end, std::back_inserter(k_indices));
#else
						std::copy_if(it_begin, it_end, std::back_inserter(k_indices), [&visited](const auto &idx) {
							//return !visited[idx];
							return 2 != visited[idx];
						});
#endif
					}
#endif
			}
		}

		if (cluster.size() >= min_pts)
			clusters.emplace_back(std::move(cluster));
	}

	// Sort clusters
#if 0
	for (auto &cluster: clusters)
		std::sort(cluster.begin(), cluster.end());
#elif 0
	std::for_each(clusters.begin(), clusters.end(), [](auto &cluster) {
		std::sort(cluster.begin(), cluster.end());
	});
#else
	// No sorting
#endif

	return clusters;
}

void dbscan_test()
{
	// Load a point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	{
		const std::string input_filepath("path/to/input.pcd");
		//const std::string input_filepath("/work/inno3d_git/data/Autofiltering_TestData_20240828/TestData/target.pcd");

		std::cout << "Loading a point cloud from " << input_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_filepath, *cloud) != 0)
		{
			std::cerr << "File not found, " << input_filepath << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "A point cloud loaded (#points = " << cloud->size()  << "): " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;
	}

	// Cluster points by DBSCAN algorithm
	std::vector<std::vector<size_t>> clusters;
	{
		std::cout << "Clustering points by DBSCAN algorithm..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		const double radius(10.0);  // The radius for searching neighbor points of octree
		const size_t min_pts(8);  // The minimum number of points required to form a dense region
		const double octree_resolution(5.0);
		clusters = dbscan(cloud, radius, min_pts, octree_resolution);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Points clustered by DBSCAN algorithm (#clusters = " << clusters.size() << "): " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;

#if 1
		const auto num_clustered_points = std::accumulate(clusters.begin(), clusters.end(), 0, [](size_t sum, const auto &cluster) { return sum + cluster.size(); });
		std::set<size_t> noisy_points;
		size_t idx = 0;
		std::generate_n(std::inserter(noisy_points, noisy_points.end()), cloud->size(), [&idx]() { return idx++; });
#if 1
		for (const auto &cluster: clusters)
			for (const auto &pt_idx: cluster)
				noisy_points.erase(pt_idx);
#else
		std::for_each(clusters.begin(), clusters.end(), [&noisy_points](const auto &cluster) {
			//noisy_points.erase(cluster.begin(), cluster.end());
			std::for_each(cluster.begin(), cluster.end(), [&noisy_points](const auto &pt_idx) {
				noisy_points.erase(pt_idx);
			});
		});
#endif

		std::cout << "#clustered points = " << num_clustered_points << std::endl;
		std::cout << "#noisy points = " << noisy_points.size() << std::endl;
		assert(cloud->size() == num_clustered_points + noisy_points.size());

		// Add a noisy cluster
		clusters.push_back(std::vector<size_t>(noisy_points.begin(), noisy_points.end()));
#endif
	}

	// Visualize
	{
#if 1
		pcl::visualization::PCLVisualizer viewer("3D Viewer");
#if 1
		viewer.setCameraPosition(
			0.0, 0.0, 1000.0,  // The coordinates of the camera location
			0.0, 0.0, 0.0,  // The components of the view point of the camera
			0.0, 1.0, 0.0  // The component of the view up direction of the camera
		);
		viewer.setCameraFieldOfView(M_PI / 2.0);  // [rad]
		viewer.setCameraClipDistances(1.0, 1000.0);
#elif 0
		pcl::PointXYZ min_point, max_point;
		pcl::getMinMax3D(*cloud, min_point, max_point);
		std::cout << "Center point of registered point cloud = (" << (min_point.x + max_point.x) / 2 << ", " << (min_point.y + max_point.y) / 2 << ", " << (min_point.z + max_point.z) / 2 << ")." << std::endl;
		viewer.setCameraPosition(
			0.0, 0.0, 1000.0,  // The coordinates of the camera location.
			(min_point.x + max_point.x) / 2.0, (min_point.y + max_point.y) / 2.0, (min_point.z + max_point.z) / 2.0,  // The components of the view point of the camera
			0.0, 1.0, 0.0  // The component of the view up direction of the camera.
		);
		viewer.setCameraFieldOfView(M_PI / 2.0);  // [rad]
		viewer.setCameraClipDistances(1.0, 1000.0);
#else
		viewer.initCameraParameters();
#endif
		viewer.setBackgroundColor(0.5, 0.5, 0.5);
		viewer.addCoordinateSystem(100.0);

		size_t cluster_idx = 0;
		for (const auto &cluster: clusters)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			cluster_cloud->reserve(cluster.size());
			for (const auto &pt_idx: cluster)
				cluster_cloud->push_back(cloud->points[pt_idx]);

#if 1
			const std::string cloud_id("Point Cloud " + std::to_string(cluster_idx++));

			//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZ> rgb(cluster_cloud);
			//viewer.addPointCloud<pcl::PointXYZ>(cluster_cloud, rgb, cloud_id);
			viewer.addPointCloud<pcl::PointXYZ>(cluster_cloud, cloud_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, cloud_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, double(std::rand() + 1) / RAND_MAX, double(std::rand() + 1) / RAND_MAX, double(std::rand() + 1) / RAND_MAX, cloud_id);
#else
			// Estimate normals
			pcl::PointCloud<pcl::Normal>::Ptr cluster_normals(new pcl::PointCloud<pcl::Normal>());
			{
				std::cout << "Estimating normals..." << std::endl;
				const auto start_time(std::chrono::high_resolution_clock::now());
				//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
				pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
				ne.setInputCloud(cluster_cloud);
				pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
				ne.setSearchMethod(tree);
				ne.setRadiusSearch(5.0);  // Use all neighbors in a sphere of radius 5m
				ne.compute(*cluster_normals);  // Compute the features
				const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
				std::cout << "Normals estimated: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;
			}

			const std::string cloud_id("Point Cloud " + std::to_string(cluster_idx)), cloud_normals_id("Point Cloud Normal " + std::to_string(cluster_idx));
			++cluster_idx;

			//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZ> rgb(cluster_cloud);
			//viewer.addPointCloud<pcl::PointXYZ>(cluster_cloud, rgb, cloud_id);
			viewer.addPointCloud<pcl::PointXYZ>(cluster_cloud, cloud_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, cloud_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, double(std::rand() + 1) / RAND_MAX, double(std::rand() + 1) / RAND_MAX, double(std::rand() + 1) / RAND_MAX, cloud_id);
			viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cluster_cloud, cluster_normals, 10, 10.0f, cloud_normals_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, cloud_normals_id);
#endif
		}

		while (!viewer.wasStopped())
		{
			viewer.spinOnce(1);
			//std::this_thread::sleep_for(1ms);
		}
#elif 0
		pcl::visualization::CloudViewer viewer("Simple 3D Viewer");
		viewer.showCloud(cloud, "Point Cloud");
		while (!viewer.wasStopped());
#else
		// Visualize nothing
#endif
	}
}

void dbscan_gpu_test()
{
	// Load a point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	{
		const std::string input_filepath("path/to/input.pcd");
		//const std::string input_filepath("/work/inno3d_git/data/Autofiltering_TestData_20240828/TestData/target.pcd");

		std::cout << "Loading a point cloud from " << input_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_filepath, *cloud) != 0)
		{
			std::cerr << "File not found, " << input_filepath << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "A point cloud loaded (#points = " << cloud->size()  << "): " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;
	}

	// Cluster points by DBSCAN algorithm
	std::vector<std::vector<size_t>> clusters;
	{
		std::cout << "Clustering points by DBSCAN (GPU) algorithm..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		const double radius(10.0);  // The radius for searching neighbor points of octree
		const size_t min_pts(8);  // The minimum number of points required to form a dense region
		clusters = dbscan_gpu(cloud, radius, min_pts);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Points clustered by DBSCAN (GPU) algorithm (#clusters = " << clusters.size() << "): " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;

#if 1
		const auto num_clustered_points = std::accumulate(clusters.begin(), clusters.end(), 0, [](size_t sum, const auto &cluster) { return sum + cluster.size(); });
		std::set<size_t> noisy_points;
		size_t idx = 0;
		std::generate_n(std::inserter(noisy_points, noisy_points.end()), cloud->size(), [&idx]() { return idx++; });
#if 1
		for (const auto &cluster: clusters)
			for (const auto &pt_idx: cluster)
				noisy_points.erase(pt_idx);
#else
		std::for_each(clusters.begin(), clusters.end(), [&noisy_points](const auto &cluster) {
			//noisy_points.erase(cluster.begin(), cluster.end());
			std::for_each(cluster.begin(), cluster.end(), [&noisy_points](const auto &pt_idx) {
				noisy_points.erase(pt_idx);
			});
		});
#endif

		std::cout << "#clustered points = " << num_clustered_points << std::endl;
		std::cout << "#noisy points = " << noisy_points.size() << std::endl;
		assert(cloud->size() == num_clustered_points + noisy_points.size());

		// Add a noisy cluster
		clusters.push_back(std::vector<size_t>(noisy_points.begin(), noisy_points.end()));
#endif
	}

	// Visualize
	{
#if 1
		pcl::visualization::PCLVisualizer viewer("3D Viewer");
#if 1
		viewer.setCameraPosition(
			0.0, 0.0, 1000.0,  // The coordinates of the camera location
			0.0, 0.0, 0.0,  // The components of the view point of the camera
			0.0, 1.0, 0.0  // The component of the view up direction of the camera
		);
		viewer.setCameraFieldOfView(M_PI / 2.0);  // [rad]
		viewer.setCameraClipDistances(1.0, 1000.0);
#elif 0
		pcl::PointXYZ min_point, max_point;
		pcl::getMinMax3D(*cloud, min_point, max_point);
		std::cout << "Center point of registered point cloud = (" << (min_point.x + max_point.x) / 2 << ", " << (min_point.y + max_point.y) / 2 << ", " << (min_point.z + max_point.z) / 2 << ")." << std::endl;
		viewer.setCameraPosition(
			0.0, 0.0, 1000.0,  // The coordinates of the camera location.
			(min_point.x + max_point.x) / 2.0, (min_point.y + max_point.y) / 2.0, (min_point.z + max_point.z) / 2.0,  // The components of the view point of the camera
			0.0, 1.0, 0.0  // The component of the view up direction of the camera.
		);
		viewer.setCameraFieldOfView(M_PI / 2.0);  // [rad]
		viewer.setCameraClipDistances(1.0, 1000.0);
#else
		viewer.initCameraParameters();
#endif
		viewer.setBackgroundColor(0.5, 0.5, 0.5);
		viewer.addCoordinateSystem(100.0);

		size_t cluster_idx = 0;
		for (const auto &cluster: clusters)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			cluster_cloud->reserve(cluster.size());
			for (const auto &pt_idx: cluster)
				cluster_cloud->push_back(cloud->points[pt_idx]);

#if 1
			const std::string cloud_id("Point Cloud " + std::to_string(cluster_idx++));

			//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZ> rgb(cluster_cloud);
			//viewer.addPointCloud<pcl::PointXYZ>(cluster_cloud, rgb, cloud_id);
			viewer.addPointCloud<pcl::PointXYZ>(cluster_cloud, cloud_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, cloud_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, double(std::rand() + 1) / RAND_MAX, double(std::rand() + 1) / RAND_MAX, double(std::rand() + 1) / RAND_MAX, cloud_id);
#else
			// Estimate normals
			pcl::PointCloud<pcl::Normal>::Ptr cluster_normals(new pcl::PointCloud<pcl::Normal>());
			{
				std::cout << "Estimating normals..." << std::endl;
				const auto start_time(std::chrono::high_resolution_clock::now());
				//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
				pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
				ne.setInputCloud(cluster_cloud);
				pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
				ne.setSearchMethod(tree);
				ne.setRadiusSearch(5.0);  // Use all neighbors in a sphere of radius 5m
				ne.compute(*cluster_normals);  // Compute the features
				const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
				std::cout << "Normals estimated: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;
			}

			const std::string cloud_id("Point Cloud " + std::to_string(cluster_idx)), cloud_normals_id("Point Cloud Normal " + std::to_string(cluster_idx));
			++cluster_idx;

			//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZ> rgb(cluster_cloud);
			//viewer.addPointCloud<pcl::PointXYZ>(cluster_cloud, rgb, cloud_id);
			viewer.addPointCloud<pcl::PointXYZ>(cluster_cloud, cloud_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, cloud_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, double(std::rand() + 1) / RAND_MAX, double(std::rand() + 1) / RAND_MAX, double(std::rand() + 1) / RAND_MAX, cloud_id);
			viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cluster_cloud, cluster_normals, 10, 10.0f, cloud_normals_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, cloud_normals_id);
#endif
		}

		while (!viewer.wasStopped())
		{
			viewer.spinOnce(1);
			//std::this_thread::sleep_for(1ms);
		}
#elif 0
		pcl::visualization::CloudViewer viewer("Simple 3D Viewer");
		viewer.showCloud(cloud, "Point Cloud");
		while (!viewer.wasStopped());
#else
		// Visualize nothing
#endif
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void segmentation()
{
	//local::euclidean_cluster_extraction_tutorial();  // pcl::SACSegmentation
	//local::euclidean_cluster_extraction_gpu_example();  // GPU. pcl::EuclideanClusterExtraction & pcl::gpu::EuclideanClusterExtraction
	//local::region_growing_segmentation_tutorial();
	//local::min_cut_segmentation_tutorial();
	//local::conditional_euclidean_clustering_tutorial();
	//local::don_segmentation_tutorial();  // Difference of normals (DoN) features
	local::supervoxels_clustering_tutorial();  // Voxel cloud connectivity segmentation (VCCS)

	//local::region_growing_example();
	//local::supervoxels_example();  // Voxel cloud connectivity segmentation (VCCS). Not yet implemented
	//local::lccp_segmentatio_example();  // Locally convex connected patches (LCCP). Not yet implemented
	//local::cpc_segmentatio_example();  // Constrained planar cuts (CPC). Not yet implemented

	//local::grabcut_test();  // Not yet implemented

	//-----
	// Clustering

	// DBSCAN
	//local::dbscan_test();
	//local::dbscan_gpu_test();  // Slow
}

}  // namespace my_pcl
