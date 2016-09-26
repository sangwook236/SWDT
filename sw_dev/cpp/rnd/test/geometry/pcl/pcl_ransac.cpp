#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <iostream>


namespace {
namespace local {

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	// Open 3D viewer and add point cloud.
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	//viewer->addCoordinateSystem (1.0, "global");
	viewer->initCameraParameters();
	return viewer;
}

void plane_estimation()
{
	// Initialize PointClouds.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr final(new pcl::PointCloud<pcl::PointXYZ>);
	const bool showOutliers = true;

	// Populate our PointCloud with points.
	cloud->width = 500;
	cloud->height = 1;
	cloud->is_dense = false;
	cloud->points.resize(cloud->width * cloud->height);
	for (size_t i = 0; i < cloud->points.size(); ++i)
	{
		cloud->points[i].x = 1024 * std::rand() / (RAND_MAX + 1.0);
		cloud->points[i].y = 1024 * std::rand() / (RAND_MAX + 1.0);
		if (0 == i % 2)
			cloud->points[i].z = 1024 * std::rand() / (RAND_MAX + 1.0);
		else
			cloud->points[i].z = -1 * (cloud->points[i].x + cloud->points[i].y);
	}

	// Created RandomSampleConsensus object and compute the appropriated model.
	std::vector<int> inliers;
	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud));
	if (!showOutliers)
	{
		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
		ransac.setDistanceThreshold(0.01);
		ransac.computeModel();
		ransac.getInliers(inliers);
	}

	// Copies all inliers of the model computed to another PointCloud.
	pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *final);

	// Creates the visualization object and adds either our orignial cloud or all of the inliers depending on the command line arguments specified.
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = showOutliers ? simpleVis(cloud) : simpleVis(final);
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void sphere_estimation()
{
	// Initialize PointClouds.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr final(new pcl::PointCloud<pcl::PointXYZ>);
	const bool showOutliers = true;

	// Populate our PointCloud with points.
	cloud->width = 500;
	cloud->height = 1;
	cloud->is_dense = false;
	cloud->points.resize(cloud->width * cloud->height);
	for (size_t i = 0; i < cloud->points.size(); ++i)
	{
		cloud->points[i].x = 1024 * std::rand() / (RAND_MAX + 1.0);
		cloud->points[i].y = 1024 * std::rand() / (RAND_MAX + 1.0);
		if (0 == i % 5)
			cloud->points[i].z = 1024 * std::rand() / (RAND_MAX + 1.0);
		else if (0 == i % 2)
			cloud->points[i].z = std::sqrt(1 - (cloud->points[i].x * cloud->points[i].x) - (cloud->points[i].y * cloud->points[i].y));
		else
			cloud->points[i].z = -std::sqrt(1 - (cloud->points[i].x * cloud->points[i].x) - (cloud->points[i].y * cloud->points[i].y));
	}

	// Created RandomSampleConsensus object and compute the appropriated model.
	std::vector<int> inliers;
	pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(cloud));
	if (!showOutliers)
	{
		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_s);
		ransac.setDistanceThreshold(0.01);
		ransac.computeModel();
		ransac.getInliers(inliers);
	}

	// Copies all inliers of the model computed to another PointCloud.
	pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *final);

	// Creates the visualization object and adds either our orignial cloud or all of the inliers depending on the command line arguments specified.
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = showOutliers ? simpleVis(cloud) : simpleVis(final);
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

// REF [site] >> http://pointclouds.org/documentation/tutorials/random_sample_consensus.php
void ransac()
{
	local::plane_estimation();
	//local::sphere_estimation();
}

}  // namespace my_pcl
