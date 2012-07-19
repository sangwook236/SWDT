#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/smart_ptr.hpp>
#include <boost/thread/thread.hpp>


namespace {
namespace local {
	
}  // namespace local
}  // unnamed namespace

void greedy_projection()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	sensor_msgs::PointCloud2 cloud_blob;
	pcl::io::loadPCDFile("./pcl_data/bun0.pcd", cloud_blob);
	pcl::fromROSMsg(cloud_blob, *cloud);
	// the data should be available in cloud

	// normal estimation
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);
	// normals should not contain the point normals + surface curvatures

	// concatenate the XYZ and normal fields
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
	// cloud_with_normals = cloud + normals

	// create search tree
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>());
	tree2->setInputCloud(cloud_with_normals);

	// initialize objects
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh triangles;

	// set the maximum distance between connected points (maximum edge length)
	gp3.setSearchRadius(0.025);

	// set typical values for the parameters
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(100);
	gp3.setMaximumSurfaceAngle(M_PI / 4);  // 45 degrees
	gp3.setMinimumAngle(M_PI / 18);  // 10 degrees
	gp3.setMaximumAngle(2 * M_PI / 3);  // 120 degrees
	gp3.setNormalConsistency(false);

	// get result
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree2);
	gp3.reconstruct(triangles);

	// additional vertex information
	std::vector<int> parts = gp3.getPartIDs();
	std::vector<int> states = gp3.getPointStates();

	// save output
	pcl::io::saveVTKFile("./pcl_data/bun0_mesh.vtk", triangles);

	//
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	viewer->addPolygonMesh(triangles, "polygon");

	//
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}