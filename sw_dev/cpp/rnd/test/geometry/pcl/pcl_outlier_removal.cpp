#include <chrono>
#include <string>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>


using namespace std::literals::chrono_literals;

namespace {
namespace local {

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/master/statistical_outlier.html
void statistical_outlier_removal_example()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
	pcl::PCDReader reader;
	reader.read<pcl::PointXYZ>("./table_scene_lms400.pcd", *cloud);

	std::cerr << "Cloud before statistical outlier removal:" << std::endl;
	std::cerr << *cloud << std::endl;

	//-----
	// Create the filtering object
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	{
		std::cout << "Performing statistical outlier removal..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		sor.setInputCloud(cloud);
		sor.setSearchMethod(tree);
		sor.setMeanK(50);
		sor.setStddevMulThresh(1.0);
		//sor.setKeepOrganized(true);
		//sor.setUserFilterValue(std::numeric_limits<float>::quiet_NaN());
		//sor.setNegative(true);
		sor.filter(*cloud_filtered);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Statistical outlier removal performed (#points = " << cloud_filtered->size() << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;
	}

	std::cerr << "Cloud after statistical outlier removal:" << std::endl;
	std::cerr << *cloud_filtered << std::endl;
}

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/master/remove_outliers.html
void radius_outlier_removal_example()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
	pcl::PCDReader reader;
	reader.read<pcl::PointXYZ>("./table_scene_lms400.pcd", *cloud);

	std::cerr << "Cloud before radius outlier removal:" << std::endl;
	std::cerr << *cloud << std::endl;

	//-----
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	{
		std::cout << "Performing radius outlier removal..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		ror.setInputCloud(cloud);
		ror.setSearchMethod(tree);
		ror.setRadiusSearch(0.8);
		ror.setMinNeighborsInRadius(2);
		//ror.setKeepOrganized(true);
		//ror.setUserFilterValue(std::numeric_limits<float>::quiet_NaN());
		//ror.setNegative(true)

		ror.filter(*cloud_filtered);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Radius outlier removal performed (#points = " << cloud_filtered->size() << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;
	}

	std::cerr << "Cloud after radius outlier removal:" << std::endl;
	std::cerr << *cloud_filtered << std::endl;
}

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/master/remove_outliers.html
void conditional_removal_example()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
	pcl::PCDReader reader;
	reader.read<pcl::PointXYZ>("./table_scene_lms400.pcd", *cloud);

	std::cerr << "Cloud before conditional removal:" << std::endl;
	std::cerr << *cloud << std::endl;

	//-----
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	{
		// Build the condition
		pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>());
		range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::GT, 0.0)));
		range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::LT, 0.8)));

		std::cout << "Performing conditional removal..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::ConditionalRemoval<pcl::PointXYZ> cr;
		cr.setInputCloud(cloud);
		cr.setCondition(range_cond);
		//cr.setKeepOrganized(true);
		//cr.setUserFilterValue(std::numeric_limits<float>::quiet_NaN());

		cr.filter(*cloud_filtered);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Conditional removal performed (#points = " << cloud_filtered->size() << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;
	}

	std::cerr << "Cloud after conditional removal:" << std::endl;
	std::cerr << *cloud_filtered << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void outlier_removal()
{
	// Local plane filtering
	//	Refer to depth_local_plane_filter_test.py

	local::statistical_outlier_removal_example();
	//local::radius_outlier_removal_example();

	//local::conditional_removal_example();
}

}  // namespace my_pcl
