#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/file_io.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_handlers.h>
#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/initialization.h>
//#include "openni_capture.h"
//#include "../src/internal.h"
#include "/internal.h"
#include "tsdf_volume.h"
#include "tsdf_volume.hpp"


namespace {
namespace local {

using PointT = pcl::PointXYZ;
using VoxelT = float;
using WeightT = short;

/** \brief Class for storing and handling the TSDF Volume on the GPU */
class DeviceVolume
{
public:
	using Ptr = std::shared_ptr<DeviceVolume>;
	using ConstPtr = std::shared_ptr<const DeviceVolume>;

	/** \brief Constructor
	 * param[in] volume_size size of the volume in mm
	 * param[in] volume_res volume grid resolution (typically device::VOLUME_X x device::VOLUME_Y x device::VOLUME_Z)
	 */
	DeviceVolume(const Eigen::Vector3f &volume_size, const Eigen::Vector3i &volume_res)
	: volume_size_(volume_size)
	{
		// Initialize GPU
		device_volume_.create(volume_res[1] * volume_res[2], volume_res[0]); //(device::VOLUME_Y * device::VOLUME_Z, device::VOLUME_X)
		pcl::device::initVolume(device_volume_);

		// Truncation distance
		Eigen::Vector3f voxel_size = volume_size.array() / volume_res.array().cast<float>();
		trunc_dist_ = std::max((float)min_trunc_dist, 2.1f * std::max(voxel_size[0], std::max(voxel_size[1], voxel_size[2])));
	};

	/** \brief Creates the TSDF volume on the GPU
	 * param[in] depth depth readings from the sensor
	 * param[in] intr camera intrinsics
	 */
	void createFromDepth(const pcl::device::PtrStepSz<const unsigned short> &depth, const pcl::device::Intr &intr);

	/** \brief Downloads the volume from the GPU
	 * param[out] volume volume structure where the data is written to (size needs to be appropriately set beforehand (is checked))
	 */
	bool getVolume(pcl::TSDFVolume<VoxelT, WeightT>::Ptr &volume);

	/** \brief Generates and returns a point cloud form the implicit surface in the TSDF volume
	 * param[out] cloud point cloud containing the surface
	 */
	bool getCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

private:
	template<class D, class Matx>
	D & device_cast(Matx &matx)
	{
		return (*reinterpret_cast<D *>(matx.data()));
	}

	pcl::gpu::DeviceArray2D<int> device_volume_;
	Eigen::Vector3f volume_size_;
	float trunc_dist_;
};

void DeviceVolume::createFromDepth(const pcl::device::PtrStepSz<const unsigned short> &depth, const pcl::device::Intr &intr)
{
	using Matrix3frm = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

	// FIXME [check] >> magic numbers
	constexpr int rows = 480;
	constexpr int cols = 640;

	// Scale depth values
	pcl::gpu::DeviceArray2D<float> device_depth_scaled;
	device_depth_scaled.create(rows, cols);

	// Upload depth map on GPU
	pcl::gpu::KinfuTracker::DepthMap device_depth;
	device_depth.upload(depth.data, depth.step, depth.rows, depth.cols);

	// Initial camera rotation and translation
	Matrix3frm init_Rcam = Eigen::Matrix3f::Identity();
	Eigen::Vector3f init_tcam = volume_size_ * 0.5f - Eigen::Vector3f(0, 0, volume_size_(2) / 2 * 1.2f);

	Matrix3frm init_Rcam_inv = init_Rcam.inverse();
	pcl::device::Mat33 &device_Rcam_inv = device_cast<pcl::device::Mat33>(init_Rcam_inv);
	float3 &device_tcam = device_cast<float3>(init_tcam);

	// Integrate depth values into volume
	float3 device_volume_size = device_cast<float3>(volume_size_);
	pcl::device::integrateTsdfVolume(device_depth, intr, device_volume_size, device_Rcam_inv, device_tcam, trunc_dist_, device_volume_, device_depth_scaled);
}

bool DeviceVolume::getVolume(pcl::TSDFVolume<VoxelT, WeightT>::Ptr &volume)
{
	int volume_size = device_volume_.rows() * device_volume_.cols();

	if ((std::size_t)volume_size != volume->size())
	{
		pcl::console::print_error("Device volume size (%d) and tsdf volume size (%d) don't match. ABORTING!\n", volume_size, volume->size());
		return false;
	}

	std::vector<VoxelT> &volume_vec = volume->volumeWriteable();
	std::vector<WeightT> &weights_vec = volume->weightsWriteable();

	device_volume_.download(&volume_vec[0], device_volume_.cols() * sizeof(int));

	#pragma omp parallel for default(none) shared(volume, volume_vec, weights_vec)
	for(int i = 0; i < (int)volume->size(); ++i)
	{
		short2 *elem = (short2 *)&volume_vec[i];
		volume_vec[i]  = float(elem->x) / pcl::device::DIVISOR;
		weights_vec[i] = short(elem->y);
	}

	return true;
}

bool DeviceVolume::getCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
	// FIXME [check] >> fixed size
	constexpr int DEFAULT_VOLUME_CLOUD_BUFFER_SIZE = 10 * 1000 * 1000;

	// Point buffer on the device
	pcl::gpu::DeviceArray<pcl::PointXYZ> device_cloud_buffer(DEFAULT_VOLUME_CLOUD_BUFFER_SIZE);

	// Do the extraction
	float3 device_volume_size = device_cast<float3>(volume_size_);
	/*size_t size =*/ pcl::device::extractCloud(device_volume_, device_volume_size, device_cloud_buffer);

	// Write into point cloud structure
	device_cloud_buffer.download(cloud->points);
	cloud->width = cloud->size();
	cloud->height = 1;

	return true;
}

#if 0
/** \brief Converts depth and RGB sensor readings into a point cloud
 * param[in] depth depth data from sensor
 * param[in] rgb24 color data from sensor
 * param[in] intr camera intrinsics
 * param[out] cloud the generated point cloud
 * \note: position in mm is converted to m
 * \note: RGB reading not working!
 */
//TODO implement correct color reading (how does rgb24 look like?)
bool convertDepthRGBToCloud(const pcl::device::PtrStepSz<const unsigned short> &depth, const pcl::device::PtrStepSz<const pcl::gpu::KinfuTracker::PixelRGB> &rgb24, const pcl::device::Intr &intr, pcl::PointCloud<PointT>::Ptr &cloud)
{
	// Resize point cloud if it doesn't fit
	if (depth.rows != (int)cloud->height || depth.cols != (int)cloud->width)
		cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>(depth.cols, depth.rows));

	//std::cout << "step = " << rgb24.step << std::endl;
	//std::cout << "elem size = " << rgb24.elem_size << std::endl;

	// Iterate over all depth and rgb values
	for (int y = 0; y < depth.rows; ++y)
	{
		// Get pointers to the values in one row
		const unsigned short *depth_row_ptr = depth.ptr(y);
		//const pcl::gpu::KinfuTracker::RGB *rgb24_row_ptr = rgb24.ptr(y);
		//const char *rgb24_row_ptr = (const char *)rgb24.ptr(y);

		// Iterate over row and store values
		for (int x = 0; x < depth.cols; ++x)
		{
			float u = (x - intr.cx) / intr.fx;
			float v = (y - intr.cy) / intr.fy;

			PointT &point = cloud->at(x, y);

			point.z = depth_row_ptr[x] / 1000.0f;
			point.x = u * point.z;
			point.y = v * point.z;

/*
			std::uint8_t r = *(rgb24_row_ptr + 0);
			std::uint8_t g = *(rgb24_row_ptr + 1);
			std::uint8_t b = *(rgb24_row_ptr + 2);
			std::uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
			point.rgb = *reinterpret_cast<float *>(&rgb);

			point.r = *((const char *)rgb24.data + y * rgb24.step + x * rgb24.elem_size);
			point.g = *((const char *)rgb24.data + y * rgb24.step + x * rgb24.elem_size + 1);
			point.b = *((const char *)rgb24.data + y * rgb24.step + x * rgb24.elem_size + 2);
*/
		}
	}

	cloud->is_dense = false;

	return true;
}

/** \brief Captures data from a sensor and generates a point cloud from it
 * param[in] capture capturing device object
 * param[out] depth the depth reading
 * param[out] rgb24 the color reading
 * param[out] intr camera intrinsics for this reading
 * param[out] cloud point cloud generated from the readings
 */
bool captureCloud(
	pcl::gpu::CaptureOpenNI &capture,
	pcl::device::PtrStepSz<const unsigned short> &depth, pcl::device::PtrStepSz<const pcl::gpu::KinfuTracker::PixelRGB> &rgb24,
	pcl::device::Intr &intr, pcl::PointCloud<PointT>::Ptr &cloud
)
{
	// Capture frame
	if (!capture.grab(depth, rgb24))
	{
		pcl::console::print_error("Can't capture via sensor.\n");
		return false;
	}

	// Get intrinsics from capture
	float f = capture.depth_focal_length_VGA;
	intr = pcl::device::Intr(f, f, depth.cols / 2, depth.rows / 2);

	// Generate point cloud
	cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>(depth.cols, depth.rows));
	if (!convertDepthRGBToCloud(depth, rgb24, intr, cloud))
	{
		pcl::console::print_error("Conversion depth --> cloud was not successful!\n");
		return false;
	}

	return true;
}
#else
/** \brief Converts depth and RGB sensor readings into a point cloud
 * param[in] depth depth data from sensor
 * param[in] rgb24 color data from sensor
 * param[out] cloud the generated point cloud
 * \note: position in mm is converted to m
 * \note: RGB reading not working!
 */
//TODO implement correct color reading (how does rgb24 look like?)
bool convertDepthRGBToCloud(const pcl::device::PtrStepSz<const unsigned short> &depth, const pcl::device::PtrStepSz<const pcl::gpu::KinfuTracker::PixelRGB> &rgb24, pcl::PointCloud<PointT>::Ptr &cloud)
{
	// Resize point cloud if it doesn't fit
	if (depth.rows != (int)cloud->height || depth.cols != (int)cloud->width)
		cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>(depth.cols, depth.rows));

	//std::cout << "step = " << rgb24.step << std::endl;
	//std::cout << "elem size = " << rgb24.elem_size << std::endl;

	// Iterate over all depth and rgb values
	for (int y = 0; y < depth.rows; ++y)
	{
		// Get pointers to the values in one row
		const unsigned short *depth_row_ptr = depth.ptr(y);
		//const pcl::gpu::KinfuTracker::RGB *rgb24_row_ptr = rgb24.ptr(y);
		//const char *rgb24_row_ptr = (const char *)rgb24.ptr(y);

		// Iterate over row and store values
		for (int x = 0; x < depth.cols; ++x)
		{
			float u = (x - intr.cx) / intr.fx;
			float v = (y - intr.cy) / intr.fy;

			PointT &point = cloud->at(x, y);

			point.z = depth_row_ptr[x] / 1000.0f;
			point.x = u * point.z;
			point.y = v * point.z;

/*
			std::uint8_t r = *(rgb24_row_ptr + 0);
			std::uint8_t g = *(rgb24_row_ptr + 1);
			std::uint8_t b = *(rgb24_row_ptr + 2);
			std::uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
			point.rgb = *reinterpret_cast<float *>(&rgb);

			point.r = *((const char *)rgb24.data + y * rgb24.step + x * rgb24.elem_size);
			point.g = *((const char *)rgb24.data + y * rgb24.step + x * rgb24.elem_size + 1);
			point.b = *((const char *)rgb24.data + y * rgb24.step + x * rgb24.elem_size + 2);
*/
		}
	}

	cloud->is_dense = false;

	return true;
}
#endif

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/gpu/kinfu/tools/record_tsdfvolume.cpp
void kinfu_record_tsdfvolume_example()
{
	const std::string cloud_file("./cloud.pcd");
	const std::string volume_file("./tsdf_volume.dat");

	const double min_trunc_dist = 30.0f;

	const bool quit = false, save = false;
	const bool extract_cloud_volume = false;

	//--------------------
	// Set up and visualize

	pcl::gpu::setDevice(0);
	pcl::gpu::printShortCudaDeviceInfo(0);

	pcl::device::PtrStepSz<const unsigned short> depth;
	pcl::device::PtrStepSz<const pcl::gpu::KinfuTracker::PixelRGB> rgb24;

#if 0
	pcl::gpu::CaptureOpenNI capture(0);  // First OpenNI device
	pcl::PointCloud<PointT>::Ptr cloud; //(new pcl::PointCloud<PointT>);
	pcl::device::Intr intr;

	// Capture first frame
	if (!captureCloud(capture, depth, rgb24, intr, cloud))
		return;
#else
	// Generate point cloud
	pcl::PointCloud<PointT>::Ptr cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>(depth.cols, depth.rows));
	if (!convertDepthRGBToCloud(depth, rgb24, cloud))
	{
		pcl::console::print_error("Conversion depth --> cloud was not successful!\n");
		return;
	}
#endif

	// Start visualizer
	pcl::visualization::PCLVisualizer visualizer;
	//pcl::visualization::PointCloudColorHandlerRGBField<PointT> color_handler(cloud);
	//pcl::visualization::PointCloudColorHandlerCustom<PointT> color_handler(cloud, 0.5, 0.5, 0.5);
	visualizer.addPointCloud<PointT>(cloud); //, color_handler, "cloud");
	visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1);
	visualizer.addCoordinateSystem(1, "global");
	visualizer.initCameraParameters();
	//visualizer.registerKeyboardCallback(keyboard_callback);
	visualizer.spinOnce();

	//--------------------
	// Capture data and generate cloud
	pcl::console::print_highlight("Capturing data ... \n");

	while (!quit && !save)
	{
#if 0
		// Capture data and convert to point cloud
		if (!captureCloud(capture, depth, rgb24, intr, cloud))
			return;
#else
		// Generate point cloud
		pcl::PointCloud<PointT>::Ptr cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>(depth.cols, depth.rows));
		if (!convertDepthRGBToCloud(depth, rgb24, cloud))
		{
			pcl::console::print_error("Conversion depth --> cloud was not successful!\n");
			return;
		}
#endif

		// Update visualization
		visualizer.updatePointCloud<PointT>(cloud); //, color_handler, "cloud");
		visualizer.spinOnce();
	}

	if (quit)
		return EXIT_SUCCESS;

	//--------------------
	// Generate volume

	// Create volume object
	pcl::TSDFVolume<VoxelT, WeightT>::Ptr volume(new pcl::TSDFVolume<VoxelT, WeightT>);
	const Eigen::Vector3i resolution(pcl::device::VOLUME_X, pcl::device::VOLUME_Y, pcl::device::VOLUME_Z);
	const Eigen::Vector3f volume_size(Eigen::Vector3f::Constant(3000));
	volume->resize(resolution, volume_size);

	DeviceVolume::Ptr device_volume(new DeviceVolume(volume->volumeSize(), volume->gridResolution()));

	// Integrate depth in device volume
	pcl::console::print_highlight("Converting depth map to volume ... ");  std::cout << std::flush;
	device_volume->createFromDepth(depth, intr);

	// Get volume from device
	if (!device_volume->getVolume(volume))
	{
		pcl::console::print_error("Coudln't get volume from device!\n");
		return;
	}
	pcl::console::print_info("done [%d voxels]\n", volume->size());

	// Generate TSDF cloud
	pcl::console::print_highlight("Generating tsdf volume cloud ... ");  std::cout << std::flush;
	pcl::PointCloud<pcl::PointXYZI>::Ptr tsdf_cloud(new pcl::PointCloud<pcl::PointXYZI>);
	volume->convertToTsdfCloud(tsdf_cloud);
	pcl::console::print_info("done [%d points]\n", tsdf_cloud->size());

	// Get cloud from volume
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_volume(new pcl::PointCloud<pcl::PointXYZ>);
	if (extract_cloud_volume)
	{
		pcl::console::print_highlight("Generating cloud from volume ... ");  std::cout << std::flush;
		if (!device_volume->getCloud(cloud_volume))
		{
			pcl::console::print_error("Cloudn't get cloud from device volume!\n");
			return;
		}
		pcl::console::print_info("done [%d points]\n", cloud_volume->size());
	}

	//--------------------
	// Store results
	pcl::console::print_highlight("Storing results:\n");

	// Point cloud
	pcl::console::print_info("Saving captured cloud to ");  pcl::console::print_value("%s", cloud_file.c_str());  pcl::console::print_info(" ... ");
	if (pcl::io::savePCDFile(cloud_file, *cloud, true) < 0)
	{
		std::cout << std::endl;
		pcl::console::print_error("Cloudn't save the point cloud to file %s.\n", cloud_file.c_str());
	}
	else
		pcl::console::print_info("done [%d points].\n", cloud->size());

	// Volume
	if (!volume->save(volume_file, true))
		pcl::console::print_error("Cloudn't save the volume to file %s.\n", volume_file.c_str());

	// TSDF point cloud
	std::string tsdf_cloud_file(pcl::getFilenameWithoutExtension(volume_file) + "_cloud.pcd");
	pcl::console::print_info("Saving volume cloud to ");  pcl::console::print_value("%s", tsdf_cloud_file.c_str());  pcl::console::print_info(" ... ");
	if (pcl::io::savePCDFile(tsdf_cloud_file, *tsdf_cloud, true) < 0)
	{
		std::cout << std::endl;
		pcl::console::print_error("Cloudn't save the volume point cloud to file %s.\n", tsdf_cloud_file.c_str());
	}
	else
		pcl::console::print_info("done [%d points].\n", tsdf_cloud->size());

	// Point cloud from volume
	if (extract_cloud_volume)
	{
		std::string cloud_volume_file(pcl::getFilenameWithoutExtension(cloud_file) + "_from_volume.pcd");
		pcl::console::print_info("Saving cloud from volume to "); pcl::console::print_value("%s", cloud_volume_file.c_str());  pcl::console::print_info(" ... ");
		if(pcl::io::savePCDFile(cloud_volume_file, *cloud_volume, true) < 0)
		{
			std::cout << std::endl;
			pcl::console::print_error("Cloudn't save the point cloud to file %s.\n", cloud_volume_file.c_str());
		}
		else
			pcl::console::print_info("done [%d points].\n", cloud_volume->size());
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void tsdf()
{
	local::kinfu_record_tsdfvolume_example();
}

}  // namespace my_pcl
