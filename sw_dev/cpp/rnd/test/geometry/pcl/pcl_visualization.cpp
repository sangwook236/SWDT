#include <iostream>
#include <thread>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>


using namespace std::literals::chrono_literals;

namespace {
namespace local {

// help.
void printUsage(const char *progName)
{
	std::cout << "\n\nUsage: "<< progName << " [options]\n\n"
		<< "Options:\n"
		<< "-------------------------------------------\n"
		<< "-h           this help\n"
		<< "-s           Simple visualisation example\n"
		<< "-r           RGB colour visualisation example\n"
		<< "-c           Custom colour visualisation example\n"
		<< "-n           Normals visualisation example\n"
		<< "-a           Shapes visualisation example\n"
		<< "-v           Viewports example\n"
		<< "\n\n";
}

pcl::visualization::PCLVisualizer::Ptr simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	// Open 3D viewer and add point cloud.
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud(cloud, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return viewer;
}

pcl::visualization::PCLVisualizer::Ptr rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
	// Open 3D viewer and add point cloud.
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return viewer;
}

pcl::visualization::PCLVisualizer::Ptr customColourVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	// Open 3D viewer and add point cloud.
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "sample cloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return viewer;
}

pcl::visualization::PCLVisualizer::Ptr normalsVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
	// Open 3D viewer and add point cloud and normals.
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 10, 0.05f, "normals");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return viewer;
}

pcl::visualization::PCLVisualizer::Ptr shapesVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
	// Open 3D viewer and add point cloud.
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	// Add shapes at cloud points.
	viewer->addLine<pcl::PointXYZRGB>(cloud->points[0], cloud->points[cloud->size() - 1], "line");
	viewer->addSphere(cloud->points[0], 0.2, 0.5, 0.5, 0.0, "sphere");
	/*
	viewer->addPlane();
	viewer->addCircle();
	viewer->addPolygon();
	viewer->addArrow();
	viewer->addCone();
	viewer->addCube();
	viewer->addCylinder();
	viewer->addEllipsoid();
	viewer->addPolygonMesh();
	viewer->addText();
	viewer->addText3D();
	*/

	// Add shapes at other locations.
	pcl::ModelCoefficients coeffs;
	coeffs.values.push_back(0.0);
	coeffs.values.push_back(0.0);
	coeffs.values.push_back(1.0);
	coeffs.values.push_back(0.0);
	viewer->addPlane(coeffs, "plane");
	coeffs.values.clear();
	coeffs.values.push_back(0.3);
	coeffs.values.push_back(0.3);
	coeffs.values.push_back(0.0);
	coeffs.values.push_back(0.0);
	coeffs.values.push_back(1.0);
	coeffs.values.push_back(0.0);
	coeffs.values.push_back(5.0);
	viewer->addCone(coeffs, "cone");

	return viewer;
}

pcl::visualization::PCLVisualizer::Ptr viewportsVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals1, pcl::PointCloud<pcl::Normal>::ConstPtr normals2)
{
	// Open 3D viewer and add point cloud and normals.
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->initCameraParameters();

	int v1(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->setBackgroundColor(0, 0, 0, v1);
	viewer->addText("Radius: 0.01", 10, 10, "v1 text", v1);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud1", v1);

	int v2(0);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
	viewer->addText("Radius: 0.1", 10, 10, "v2 text", v2);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(cloud, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, single_color, "sample cloud2", v2);

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud1");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");
	viewer->addCoordinateSystem(1.0);

	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals1, 10, 0.05f, "normals1", v1);
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals2, 10, 0.05f, "normals2", v2);

	return viewer;
}

pcl::visualization::PCLVisualizer::Ptr meshesVis_xyz(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_normal(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	ne.setSearchMethod(tree_normal);
	ne.setRadiusSearch(0.2);
	ne.compute(*normals);

	// Add polygon meshes at cloud points.
	// Concatenate the XYZ and normal fields.
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
	//*cloud_with_normals = *cloud + *normals;

	// Create search tree.
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree_gp3(new pcl::search::KdTree<pcl::PointNormal>());
	tree_gp3->setInputCloud(cloud_with_normals);

	// Initialize objects.
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh());
	// Set the maximum distance between connected points (maximum edge length).
	gp3.setSearchRadius(0.2);
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(100);
	gp3.setMaximumSurfaceAngle(M_PI / 4);  // 45 degrees.
	gp3.setMinimumAngle(M_PI / 18);  // 10 degrees.
	gp3.setMaximumAngle(2 * M_PI / 3);  // 120 degrees.
	gp3.setNormalConsistency(false);
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree_gp3);
	gp3.reconstruct(*triangles);

	// Open 3D viewer and add point cloud.
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addPolygonMesh(*triangles, "polygon");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	return viewer;
}

pcl::visualization::PCLVisualizer::Ptr meshesVis_xyzrgb(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_normal(new pcl::search::KdTree<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	ne.setSearchMethod(tree_normal);
	ne.setRadiusSearch(0.2);
	ne.compute(*normals);

	// Add polygon meshes at cloud points.
	// Concatenate the XYZ and normal fields.
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
	//*cloud_with_normals = *cloud + *normals;

	// Create search tree.
	//pcl::KdTree<pcl::PointXYZRGBNormal>::Ptr tree_gp3(new pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>);
	pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree_gp3(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	tree_gp3->setInputCloud(cloud_with_normals);

	// Initialize objects.
	pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
	pcl::PolygonMesh triangles;
	// Set the maximum distance between connected points (maximum edge length).
	gp3.setSearchRadius(0.2);
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(100);
	gp3.setMaximumSurfaceAngle(M_PI / 4);  // 45 degrees.
	gp3.setMinimumAngle(M_PI / 18);  // 10 degrees.
	gp3.setMaximumAngle(2 * M_PI / 3);  // 120 degrees.
	gp3.setNormalConsistency(false);
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree_gp3);
	gp3.reconstruct(triangles);

	// Open 3D viewer and add point cloud.
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addPolygonMesh(triangles, "polygon");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	return viewer;
}

unsigned int text_id = 0;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *viewer_void)
{
	pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);
	if (event.getKeySym() == "r" && event.keyDown())
	{
		std::cout << "r was pressed => removing all text" << std::endl;

		char str[512];
		for (unsigned int i = 0; i < text_id; ++i)
		{
			sprintf(str, "text#%03d", i);
			viewer->removeShape(str);
		}
		text_id = 0;
	}
}

void mouseEventOccurred(const pcl::visualization::MouseEvent &event, void *viewer_void)
{
	pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);
	if (event.getButton() == pcl::visualization::MouseEvent::LeftButton &&
		event.getType() == pcl::visualization::MouseEvent::MouseButtonRelease)
	{
		std::cout << "Left mouse button released at position (" << event.getX() << ", " << event.getY() << ")" << std::endl;

		char str[512];
		sprintf(str, "text#%03d", text_id++);
		viewer->addText("clicked here", event.getX(), event.getY(), str);
	}
}

pcl::visualization::PCLVisualizer::Ptr interactionCustomizationVis()
{
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);

	viewer->registerKeyboardCallback(keyboardEventOccurred, (void *)viewer.get());
	viewer->registerMouseCallback(mouseEventOccurred, (void *)viewer.get());

	return viewer;
}

// REF [site] >> http://pointclouds.org/documentation/tutorials/visualization.php
void visualization_tutorial(int argc, char **argv)
{
#if 0
	// Parse Command Line Arguments.
	if (pcl::console::find_argument(argc, argv, "-h") >= 0)
	{
		local::printUsage(argv[0]);
		return;
	}

	bool simple(false), rgb(false), custom_c(false), normals(false), shapes(false), viewports(false);

	if (pcl::console::find_argument(argc, argv, "-s") >= 0)
	{
		simple = true;
		std::cout << "Simple visualisation example" << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-c") >= 0)
	{
		custom_c = true;
		std::cout << "Custom colour visualisation example" << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-r") >= 0)
	{
		rgb = true;
		std::cout << "RGB colour visualisation example" << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-n") >= 0)
	{
		normals = true;
		std::cout << "Normals visualisation example" << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-a") >= 0)
	{
		shapes = true;
		std::cout << "Shapes visualisation example" << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-v") >= 0)
	{
		viewports = true;
		std::cout << "Viewports example" << std::endl;
	}
	else
	{
		local::printUsage(argv[0]);
		return;
	}
#else
	const bool simple(true), rgb(false), custom_c(false), normals(false), shapes(false), viewports(false), meshes(false), interaction_customization(false);
#endif

	// Create example point cloud.
	pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
	std::cout << "Genarating example point clouds." << std::endl << std::endl;
	// We're going to make an ellipse extruded along the z-axis. The colour for the XYZRGB cloud will gradually go from red to green to blue.
	uint8_t r(255), g(15), b(15);
	for (float z = -1.0f; z <= 1.0f; z += 0.05f)
	{
		for (float angle = 0.0f; angle <= 360.0f; angle += 5.0f)
		{
			pcl::PointXYZ basic_point;
			//basic_point.x = 0.5f * std::cosf(pcl::deg2rad(angle));
			basic_point.x = 0.5f * std::cos(pcl::deg2rad(angle));
			//basic_point.y = std::sinf(pcl::deg2rad(angle));
			basic_point.y = std::sin(pcl::deg2rad(angle));
			basic_point.z = z;
			basic_cloud_ptr->push_back(basic_point);

			pcl::PointXYZRGB point;
			point.x = basic_point.x;
			point.y = basic_point.y;
			point.z = basic_point.z;
			uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
			point.rgb = *reinterpret_cast<float *>(&rgb);
			point_cloud_ptr->push_back(point);
		}
		if (z < 0.0)
		{
			r -= 12;
			g += 12;
		}
		else
		{
			g -= 12;
			b += 12;
		}
	}
	basic_cloud_ptr->width = basic_cloud_ptr->points.size();
	basic_cloud_ptr->height = 1;
	point_cloud_ptr->width = point_cloud_ptr->points.size();
	point_cloud_ptr->height = 1;

	// Calculate surface normals with a search radius of 0.05.
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	ne.setInputCloud(point_cloud_ptr);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	ne.setSearchMethod(tree);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1(new pcl::PointCloud<pcl::Normal>());
	ne.setRadiusSearch(0.05);
	ne.compute(*cloud_normals1);

	// Calculate surface normals with a search radius of 0.1.
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>());
	ne.setRadiusSearch(0.1);
	ne.compute(*cloud_normals2);

	pcl::visualization::PCLVisualizer::Ptr viewer;
	if (simple)
	{
		viewer = local::simpleVis(basic_cloud_ptr);
	}
	else if (rgb)
	{
		viewer = local::rgbVis(point_cloud_ptr);
	}
	else if (custom_c)
	{
		viewer = local::customColourVis(basic_cloud_ptr);
	}
	else if (normals)
	{
		viewer = local::normalsVis(point_cloud_ptr, cloud_normals2);
	}
	else if (shapes)
	{
		viewer = local::shapesVis(point_cloud_ptr);
	}
	else if (meshes)
	{
		//viewer = local::meshesVis_xyz(basic_cloud_ptr);
		viewer = local::meshesVis_xyzrgb(point_cloud_ptr);
	}
	else if (viewports)
	{
		viewer = local::viewportsVis(point_cloud_ptr, cloud_normals1, cloud_normals2);
	}
	else if (interaction_customization)
	{
		viewer = local::interactionCustomizationVis();
	}

	// Main loop.
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		std::this_thread::sleep_for(100ms);
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void visualization(int argc, char **argv)
{
	// REF [function] >> visualize_point_clouds() in ${NGVTECH_HOME}/cpp/test/pcl_test/main.cpp

	local::visualization_tutorial(argc, argv);
}

}  // namespace my_pcl
