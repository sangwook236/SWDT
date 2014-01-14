//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/circular_buffer.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>


namespace my_opencv {

// [ref] ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp.
void draw_histogram_1D(const cv::MatND &hist, const int binCount, const double maxVal, const int binWidth, const int maxHeight, cv::Mat &histImg);
void normalize_histogram(cv::MatND &hist, const double factor);

void segment_motion_using_mhi(const double timestamp, const double mhiTimeDuration, const cv::Mat &prev_gray_img, const cv::Mat &curr_gray_img, cv::Mat &mhi, cv::Mat &processed_mhi, cv::Mat &component_label_map, std::vector<cv::Rect> &component_rects);

}  // namespace my_opencv

namespace {
namespace local {

bool apply_frequency_analysis(const boost::circular_buffer<double> &data_buf, const double &Fs, const double &mag_threshold, const std::pair<double, double> &freq_range)
{
	const size_t L = data_buf.size();
	const cv::Mat &x = cv::Mat(std::vector<double>(data_buf.begin(), data_buf.end())).t();

	// compute the size of DFT transform
#if 1
	// FIXME [check] >>
	//const cv::Size dftSize(cv::getOptimalDFTSize(x.cols - 1), 1);
	const cv::Size dftSize(cv::getOptimalDFTSize(x.cols), 1);
#else
	// 2^n >= L ==> n = log2(L)
	const int &nn = size_t(std::ceil(std::log(double(L)) / std::log(2.0)));
	const cv::Size dftSize(cvRound(std::pow(2.0, nn)), 1);
#endif

	// allocate temporary buffers and initialize them with 0's
	cv::Mat temp_x(dftSize, x.type(), cv::Scalar::all(0));

	cv::Mat x_roi(temp_x, cv::Rect(0, 0, x.cols, x.rows));
	x.copyTo(x_roi);

	cv::Mat X;
	//cv::dft(x, X, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT, x.cols);
	cv::dft(temp_x, X, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT, x.cols);

	std::vector<cv::Mat> real_imag;
	cv::split(X, real_imag);

	cv::Mat mag, phase;
	cv::magnitude(real_imag[0], real_imag[1], mag);
	//cv::phase(real_imag[0], real_imag[1], phase);

	// FIXME [check] >>
	// available frequency range: [0, Fs / 2] ==> 0.5 * Fs * [0, 1]
	const size_t L2 = dftSize.width / 2;

	bool result = false;
	const cv::Mat thresholded_mag = mag > mag_threshold;
	for (size_t i = 0; i <= L2; ++i)
	{
		if (thresholded_mag.at<unsigned char>(0,i))
		{
			const double &freq = double(i) / L2 * 0.5 * Fs;
			//const double &freq = double(i) / (L2 + 1) * 0.5 * Fs;
			if (freq_range.first <= freq && freq <= freq_range.second) result = true;
			std::cout << "(" << freq << "," << mag.at<double>(0,i) << "), ";
		}
	}
	std::cout << std::endl;

	return result;
}

void gesture_recognition_by_frequency(cv::VideoCapture &capture)
{
	const std::string windowName("gesture recognition by frequency");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	const size_t MAX_COUNT_BUF = 1000;
	boost::circular_buffer<double> pixel_count_buf(MAX_COUNT_BUF);
	boost::circular_buffer<double> pixel_count_variation_buf(MAX_COUNT_BUF);

	// FIXME [delete] >>
	const size_t MAX_DATA_COUNT = 1000;
	std::vector<double> pixel_count_times;
	std::vector<double> mhi_pixel_counts, mhi_pixel_count_variations;
	std::vector<double> silh_pixel_counts, silh_pixel_count_variations;
	pixel_count_times.reserve(MAX_DATA_COUNT);
	mhi_pixel_counts.reserve(MAX_DATA_COUNT);
	mhi_pixel_count_variations.reserve(MAX_DATA_COUNT);
	silh_pixel_counts.reserve(MAX_DATA_COUNT);
	silh_pixel_count_variations.reserve(MAX_DATA_COUNT);
	size_t data_count = 0;
	size_t silh_last_count = 0;

	const double MHI_TIME_DURATION = 1.0;

	cv::Mat prevgray, gray, frame, frame2;
	cv::Mat mhi, img;
	size_t frame_count = 0;
	const double startTime = (double)cv::getTickCount();
	for (;;)
	{
		const double timestamp = (double)std::clock() / CLOCKS_PER_SEC;  // get current time in seconds

#if 1
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}
#else
		capture >> frame2;
		if (frame2.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}

		if (frame2.cols != imageWidth || frame2.rows != imageHeight)
		{
			//cv::resize(frame2, frame, cv::Size(imageWidth, imageHeight), 0.0, 0.0, cv::INTER_LINEAR);
			cv::pyrDown(frame2, frame);
		}
		else frame = frame2;
#endif

		cv::cvtColor(frame, gray, CV_BGR2GRAY);
		cv::cvtColor(gray, img, CV_GRAY2BGR);

		// smoothing
#if 0
		// METHOD #1: down-scale and up-scale the image to filter out the noise.

		{
			cv::Mat tmp;
			cv::pyrDown(gray, tmp);
			cv::pyrUp(tmp, gray);
		}
#elif 0
		// METHOD #2: Gaussian filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int kernelSize = 3;
			const double sigma = 2.0;
			cv::GaussianBlur(gray, gray, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);
		}
#elif 0
		// METHOD #3: box filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int ddepth = -1;  // the output image depth. -1 to use src.depth().
			const int kernelSize = 5;
			const bool normalize = true;
			cv::boxFilter(gray.clone(), gray, ddepth, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), normalize, cv::BORDER_DEFAULT);
			//cv::blur(gray.clone(), gray, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), cv::BORDER_DEFAULT);  // use the normalized box filter.
		}
#elif 0
		// METHOD #4: bilateral filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int diameter = -1;  // diameter of each pixel neighborhood that is used during filtering. if it is non-positive, it is computed from sigmaSpace.
			const double sigmaColor = 3.0;  // for range filter.
			const double sigmaSpace = 50.0;  // for space filter.
			cv::bilateralFilter(gray.clone(), gray, diameter, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
		}
#else
		// METHOD #5: no filtering.

		//gray = gray;
#endif

		// FIXME [delete] >>
		++frame_count;
		if (frame_count < 50) continue;

		if (!prevgray.empty())
		{
			if (mhi.empty())
				mhi.create(gray.rows, gray.cols, CV_32F);

			cv::Mat processed_mhi, component_label_map;
			std::vector<cv::Rect> component_rects;
			my_opencv::segment_motion_using_mhi(timestamp, MHI_TIME_DURATION, prevgray, gray, mhi, processed_mhi, component_label_map, component_rects);

			//const double flow_mag_threshold = 1.0;
			//for (std::vector<cv::Rect>::const_iterator it = component_rects.begin(); it != component_rects.end(); ++it)
			//{
			//	cv::Mat flow;
			//	cv::calcOpticalFlowFarneback(prevgray(*it), gray(*it), flow, 0.5, 3, 15, 3, 5, 1.2, 0);

			//	std::vector<cv::Mat> flows;
			//	cv::split(flow, flows);

			//	cv::Mat flow_mag, flow_phase;
			//	cv::magnitude(flows[0], flows[1], flow_mag);  // return type: CV_32F
			//	//cv::phase(flows[0], flows[1], flow_phase);  // return type: CV_32F

			//	component_label_map(*it).setTo(cv::Scalar::all(0), flow_mag <= flow_mag_threshold);
			//}

			//img = processed_mhi;
			cv::cvtColor(component_label_map > 0, img, CV_GRAY2BGR);
			img.setTo(cv::Scalar(255,0,0), processed_mhi >= (timestamp - 1.0e-20));  // last silhouette

			size_t k = 1;
			double min_dist = std::numeric_limits<double>::max();
			cv::Rect selected_rect;
			const double center_x = img.rows * 0.5, center_y = img.cols * 0.5;
			for (std::vector<cv::Rect>::const_iterator it = component_rects.begin(); it != component_rects.end(); ++it, ++k)
			{
				// reject very small components
				if (it->area() < 100 || it->width + it->height < 100)
					continue;

				// check for the case of little motion
				const size_t count = (size_t)cv::countNonZero((component_label_map == k)(*it));
				if (count < it->width * it->height * 0.05)
					continue;

				cv::rectangle(img, it->tl(), it->br(), CV_RGB(63, 0, 0), 2, 8, 0);

				const double x = it->x + it->width * 0.5, y = it->y + it->height * 0.5;
				const double dist = (x - center_x)*(x - center_x) + (y - center_y)*(y - center_y);
				if (dist < min_dist)
				{
					min_dist = dist;
					selected_rect = *it;
				}
			}

			if (selected_rect.area() > 0)
			{
				cv::rectangle(img, selected_rect.tl(), selected_rect.br(), CV_RGB(255, 0, 0), 2, 8, 0);

				// FIXME [delete] >>
				const int &width = selected_rect.width;
				const int &height = std::min(selected_rect.height, width / 2);
				const cv::Rect selected_rect_upper(selected_rect.x, selected_rect.y, width, height);
				cv::rectangle(img, selected_rect_upper.tl(), selected_rect_upper.br(), CV_RGB(0, 255, 0), 2, 8, 0);

				const double timespan = 0.3;  // [sec]
				const cv::Mat &img_roi = processed_mhi(selected_rect_upper) > (timestamp - timespan);
				const size_t &count = (size_t)cv::countNonZero(img_roi);

				// FIXME [delete] >>
				const cv::Mat &last_silh = processed_mhi(selected_rect_upper) >= (timestamp - 1.0e-20);
				const size_t &silh_count = (size_t)cv::countNonZero(last_silh);

				if (pixel_count_buf.empty())
				{
					pixel_count_variation_buf.push_back(0);

					// FIXME [delete] >>
					mhi_pixel_count_variations.push_back(0);
					silh_pixel_count_variations.push_back(0);
				}
				else
				{
					const double &last_count = pixel_count_buf.back();
					const double &last_variation_val = pixel_count_variation_buf.back();
					pixel_count_variation_buf.push_back(last_count > count ? (last_variation_val - 1) : (last_count < count ? (last_variation_val + 1) : last_variation_val));

					// FIXME [delete] >>
					mhi_pixel_count_variations.push_back(last_count > count ? (last_variation_val - 1) : (last_count < count ? (last_variation_val + 1) : last_variation_val));
					const double &silh_last_variation_val = silh_pixel_count_variations.back();
					silh_pixel_count_variations.push_back(silh_last_count > silh_count ? (silh_last_variation_val - 1) : (silh_last_count < silh_count ? (silh_last_variation_val + 1) : silh_last_variation_val));
				}

				pixel_count_buf.push_back(count);

				// FIXME [delete] >>
				mhi_pixel_counts.push_back(count);
				silh_pixel_counts.push_back(silh_count);
				silh_last_count = silh_count;

				//
				const double dT = ((double)cv::getTickCount() - startTime) / cv::getTickFrequency();
				const double Fs = frame_count / dT;  // sampling frequency
				const std::pair<double, double> freq_range(2.0, 3.0);

				// FIXME [delete] >>
				pixel_count_times.push_back(dT);

/*
				if (pixel_count_buf.size() > 20)
				{
					const double mag_threshold1 = 10000.0;
					std::cout << "1 ==> ";
					const bool &result1 = apply_frequency_analysis(pixel_count_buf, Fs, mag_threshold1, freq_range);
					const double mag_threshold2 = 3.0;
					std::cout << "2 ==> ";
					const bool &result2 = apply_frequency_analysis(pixel_count_variation_buf, Fs, mag_threshold2, freq_range);

					std::ostringstream sstream;
					sstream << (result1 ? "Y" : "N") << ", " << (result2 ? "Y" : "N");
					cv::putText(img, sstream.str(), cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(255, 0, 255), 1, 8, false);
				}
*/
			}

			cv::imshow(windowName, img);
		}

		// FIXME [delete] >>
		++data_count;
		if (0 == data_count % 50)
			std::cout << data_count << std::endl;
		if (data_count >= MAX_DATA_COUNT)
			break;

		if (cv::waitKey(1) >= 0)
			break;

		std::swap(prevgray, gray);
	}

	// FIXME [delete] >>
	{
		std::cout << "sz: " << pixel_count_times.size() << std::endl;
		std::cout << "sz: " << mhi_pixel_counts.size() << std::endl;
		std::cout << "sz: " << mhi_pixel_count_variations.size() << std::endl;
		std::cout << "sz: " << silh_pixel_counts.size() << std::endl;
		std::cout << "sz: " << silh_pixel_count_variations.size() << std::endl;

		std::ofstream stream("left_right.txt");
		const size_t &sz = std::min(MAX_DATA_COUNT, pixel_count_times.size());
		for (size_t i = 0; i < sz; ++i)
			stream << pixel_count_times[i] << ", " << mhi_pixel_counts[i] << ", " << mhi_pixel_count_variations[i] << ", " << silh_pixel_counts[i] << ", " << silh_pixel_count_variations[i] << std::endl;
	}

	cv::destroyWindow(windowName);
}

//-----------------------------------------------------------------------------
//

class HistogramGeneratorBase
{
protected:
	HistogramGeneratorBase()  {}
	virtual ~HistogramGeneratorBase();

public:
	virtual void createHistograms(const size_t binNum, const double histogramNormalizationFactor) = 0;

	const std::vector<cv::MatND> & getHistograms() const  {  return histograms_;  }
	const cv::MatND & getHistogram(const size_t idx) const  {  return histograms_[idx];  }

protected:
	std::vector<cv::MatND> histograms_;
};

HistogramGeneratorBase::~HistogramGeneratorBase()
{
}

//-----------------------------------------------------------------------------
//

class ReferenceHistogramGenerator: public HistogramGeneratorBase
{
public:
	// TODO [adjust] >> design parameter
	static const size_t REF_UNIMODAL_HISTOGRAM_NUM = 36;
	static const size_t REF_BIMODAL_HISTOGRAM_NUM = 0; //18;
	static const size_t REF_UNIFORM_HISTOGRAM_NUM = 0; //1;
	static const size_t REF_HISTOGRAM_NUM = REF_UNIMODAL_HISTOGRAM_NUM + REF_BIMODAL_HISTOGRAM_NUM + REF_UNIFORM_HISTOGRAM_NUM;

public:
	typedef HistogramGeneratorBase base_type;

public:
	ReferenceHistogramGenerator(const double sigma)
	: base_type(), sigma_(sigma)
	{}

private:
	ReferenceHistogramGenerator(const ReferenceHistogramGenerator &rhs);
	ReferenceHistogramGenerator & operator=(const ReferenceHistogramGenerator &rhs);

public:
	/*virtual*/ void createHistograms(const size_t binNum, const double histogramNormalizationFactor);

private:
	void createNormalHistogram(const size_t mu_idx, const double sigma, cv::MatND &hist) const;
	void createUniformHistogram(cv::MatND &hist) const;

private:
	const double sigma_;
};

/*virtual*/ void ReferenceHistogramGenerator::createHistograms(const size_t binNum, const double histogramNormalizationFactor)
{
	// create reference histograms
	histograms_.reserve(REF_HISTOGRAM_NUM);

	// unimodal distribution
	if (REF_UNIMODAL_HISTOGRAM_NUM > 0)
	{
		const size_t ref_unimodal_histogram_bin_width = (size_t)cvRound(360.0 / REF_UNIMODAL_HISTOGRAM_NUM);
		for (size_t i = 0; i < REF_UNIMODAL_HISTOGRAM_NUM; ++i)
		{
			cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
			createNormalHistogram(ref_unimodal_histogram_bin_width * i, sigma_, tmp_hist);

			// normalize histogram
			my_opencv::normalize_histogram(tmp_hist, histogramNormalizationFactor);

			histograms_.push_back(tmp_hist);
		}
	}
	// bi-modal distribution
	if (REF_BIMODAL_HISTOGRAM_NUM > 0)
	{
		//const size_t ref_bimodal_histogram_bin_width = (size_t)cvRound(180.0 / REF_BIMODAL_HISTOGRAM_NUM);
		const size_t ref_bimodal_histogram_bin_width = (size_t)cvRound(180.0 / (REF_BIMODAL_HISTOGRAM_NUM ? REF_BIMODAL_HISTOGRAM_NUM : 1));
		for (size_t i = 0; i < REF_BIMODAL_HISTOGRAM_NUM; ++i)
		{
			cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
			createNormalHistogram(ref_bimodal_histogram_bin_width * i, sigma_, tmp_hist);
			createNormalHistogram(ref_bimodal_histogram_bin_width * i + 180, sigma_, tmp_hist);

			// normalize histogram
			my_opencv::normalize_histogram(tmp_hist, histogramNormalizationFactor);

			histograms_.push_back(tmp_hist);
		}
	}
	// uniform distribution
	if (REF_UNIFORM_HISTOGRAM_NUM > 0)
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createUniformHistogram(tmp_hist);

		// normalize histogram
		my_opencv::normalize_histogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
}

void ReferenceHistogramGenerator::createNormalHistogram(const size_t mu_idx, const double sigma, cv::MatND &hist) const
{
	boost::math::normal dist(0.0, sigma);

#if 0
	for (int i = -180; i < 180; ++i)
	{
		const int kk = i + (int)mu_idx;
		const int &idx = kk >= 0 ? (kk % 360) : (360 + kk);
		hist.at<float>(idx) += (float)boost::math::pdf(dist, i);
	}
#else
	float *binPtr = (float *)hist.data;
	for (int i = -180; i < 180; ++i)
	{
		const int kk = i + (int)mu_idx;
		const int &idx = kk >= 0 ? (kk % 360) : (360 + kk);
		binPtr[idx] += (float)boost::math::pdf(dist, i);
	}
#endif
}

void ReferenceHistogramGenerator::createUniformHistogram(cv::MatND &hist) const
{
#if 0
	for (int i = 0; i < 360; ++i)
		hist.at<float>(i) += 1.0f / 360.0f;
#else
	float *binPtr = (float *)hist.data;
	for (int i = 0; i < 360; ++i, ++binPtr)
		*binPtr += 1.0f / 360.0f;
#endif
}

//-----------------------------------------------------------------------------
//

class GestureIdPatternHistogramGenerator: public HistogramGeneratorBase
{
public:
	typedef HistogramGeneratorBase base_type;

public:
	GestureIdPatternHistogramGenerator(const double sigma)
	: base_type(), sigma_(sigma)
	{}

private:
	GestureIdPatternHistogramGenerator(const GestureIdPatternHistogramGenerator &rhs);
	GestureIdPatternHistogramGenerator & operator=(const GestureIdPatternHistogramGenerator &rhs);

public:
	/*virtual*/ void createHistograms(const size_t binNum, const double histogramNormalizationFactor);

private:
	void createNormalHistogram(const size_t mu_idx, const double sigma, cv::MatND &hist) const;
	void createUniformHistogram(cv::MatND &hist) const;

private:
	const double sigma_;
};

/*virtual*/ void GestureIdPatternHistogramGenerator::createHistograms(const size_t binNum, const double histogramNormalizationFactor)
{
	// uniform distribution: for undefined gesture
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createUniformHistogram(tmp_hist);

		// normalize histogram
		my_opencv::normalize_histogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
	// unimodal distribution: for left move & left fast move
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(18, sigma_, tmp_hist);

		// normalize histogram
		my_opencv::normalize_histogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
	// unimodal distribution: for right move & right fast move
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(0, sigma_, tmp_hist);

		// normalize histogram
		my_opencv::normalize_histogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
	// unimodal distribution: for up move
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(27, sigma_, tmp_hist);

		// normalize histogram
		my_opencv::normalize_histogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
	// unimodal distribution: for down move
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(9, sigma_, tmp_hist);

		// normalize histogram
		my_opencv::normalize_histogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
	// bimodal distribution: for horizontal flip
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(0, sigma_, tmp_hist);
		createNormalHistogram(18, sigma_, tmp_hist);

		// normalize histogram
		my_opencv::normalize_histogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
	// bimodal distribution: for vertical flip
	{
		cv::MatND tmp_hist(cv::Mat::zeros(binNum, 1, CV_32FC1));
		createNormalHistogram(9, sigma_, tmp_hist);
		createNormalHistogram(27, sigma_, tmp_hist);

		// normalize histogram
		my_opencv::normalize_histogram(tmp_hist, histogramNormalizationFactor);

		histograms_.push_back(tmp_hist);
	}
}

void GestureIdPatternHistogramGenerator::createNormalHistogram(const size_t mu_idx, const double sigma, cv::MatND &hist) const
{
	boost::math::normal dist(0.0, sigma);

#if 0
	const int halfrows = hist.rows / 2;
	for (int i = -halfrows; i < halfrows; ++i)
	{
		const int kk = i + (int)mu_idx;
		const int &idx = kk >= 0 ? (kk % hist.rows) : (hist.rows + kk);
		hist.at<float>(idx) += (float)boost::math::pdf(dist, i);
	}
#else
	const int halfrows = hist.rows / 2;
	float *binPtr = (float *)hist.data;
	for (int i = -halfrows; i < halfrows; ++i)
	{
		const int kk = i + (int)mu_idx;
		const int &idx = kk >= 0 ? (kk % hist.rows) : (hist.rows + kk);
		binPtr[idx] += (float)boost::math::pdf(dist, i);
	}
#endif
}

void GestureIdPatternHistogramGenerator::createUniformHistogram(cv::MatND &hist) const
{
#if 0
	for (int i = 0; i < hist.rows; ++i)
		hist.at<float>(i) += 1.0f / (float)hist.rows;
#else
	float *binPtr = (float *)hist.data;
	for (int i = 0; i < hist.rows; ++i, ++binPtr)
		*binPtr += 1.0f / (float)hist.rows;
#endif
}

//-----------------------------------------------------------------------------
//

class HistogramAccumulator
{
public:
	//typedef HistogramAccumulator base_type;

public:
	HistogramAccumulator(const size_t histogramNum);
	HistogramAccumulator(const size_t histogramNum, const std::vector<float> &weights);

private:
	HistogramAccumulator(const HistogramAccumulator &rhs);
	HistogramAccumulator & operator=(const HistogramAccumulator &rhs);

public:
	void addHistogram(const cv::MatND &hist)  {  histograms_.push_back(hist);  }
	void clearAllHistograms()  {  histograms_.clear();  }

	size_t getHistogramSize() const  {  return histograms_.size();  }
	bool isFull() const  {  return histograms_.full();  }

	cv::MatND & getAccumulatedHistogram()  {  return accumulatedHistogram_;  }
	const cv::MatND & getAccumulatedHistogram() const  {  return accumulatedHistogram_;  }

	void accumulateHistograms();

private:
	const size_t histogramNum_;
	const std::vector<float> weights_;

	boost::circular_buffer<cv::MatND> histograms_;
	cv::MatND accumulatedHistogram_;
};

HistogramAccumulator::HistogramAccumulator(const size_t histogramNum)
: histogramNum_(histogramNum), weights_(), histograms_(histogramNum_), accumulatedHistogram_()
{
}

HistogramAccumulator::HistogramAccumulator(const size_t histogramNum, const std::vector<float> &weights)
: histogramNum_(histogramNum), weights_(weights), histograms_(histogramNum_), accumulatedHistogram_()
{
	if (weights_.size() != histogramNum_)
		throw std::runtime_error("the size of the 2nd parameter is mismatched");
}

void HistogramAccumulator::accumulateHistograms()
{
#if defined(__GNUC__)
	accumulatedHistogram_ = cv::MatND();
#else
	std::swap(accumulatedHistogram_, cv::MatND());
#endif

	if (weights_.empty())
	{
		// simple running averaging
		for (boost::circular_buffer<cv::MatND>::const_iterator it = histograms_.begin(); it != histograms_.end(); ++it)
		{
			if (accumulatedHistogram_.empty()) accumulatedHistogram_ = *it;
			else accumulatedHistogram_ += *it;
		}
	}
	else
	{
		// weighted averaging
		size_t step = 0;
		for (boost::circular_buffer<cv::MatND>::const_reverse_iterator rit = histograms_.rbegin(); rit != histograms_.rend(); ++rit, ++step)
		{
			if (accumulatedHistogram_.empty()) accumulatedHistogram_ = (*rit) * weights_[step];
			else accumulatedHistogram_ += (*rit) * weights_[step];
		}
	}
}

//-----------------------------------------------------------------------------
//

struct HistogramMatcher
{
	static size_t match(const std::vector<cv::MatND> &refHistograms, const cv::MatND &hist, double &minDist);
};

/*static*/ size_t HistogramMatcher::match(const std::vector<cv::MatND> &refHistograms, const cv::MatND &hist, double &minDist)
{
	std::vector<double> dists;
	dists.reserve(refHistograms.size());
	for (std::vector<cv::MatND>::const_iterator it = refHistograms.begin(); it != refHistograms.end(); ++it)
		dists.push_back(cv::compareHist(hist, *it, CV_COMP_BHATTACHARYYA));

	std::vector<double>::iterator itMin = std::min_element(dists.begin(), dists.end());
	minDist = *itMin;
	return (size_t)std::distance(dists.begin(), itMin);
}

//-----------------------------------------------------------------------------
//

class GestureClassifier
{
public:
	//typedef GestureClassifier base_type;

public:
	enum GestureType
	{
		GT_UNDEFINED = 0,
		GT_LEFT_MOVE, GT_RIGHT_MOVE, GT_UP_MOVE, GT_DOWN_MOVE,
		GT_LEFT_FAST_MOVE, GT_RIGHT_FAST_MOVE,
		GT_HORIZONTAL_FLIP, GT_VERTICAL_FLIP,
		GT_JAMJAM,
		GT_LEFT_90_TURN, GT_RIGHT_90_TURN,
		GT_CW, CT_CCW,
		GT_INFINITY, GT_TRIANGLE
	};

public:
	GestureClassifier(const size_t binNum, const double histogramNormalizationFactor);

private:
	GestureClassifier(const GestureClassifier &rhs);
	GestureClassifier & operator=(const GestureClassifier &rhs);

public:
	GestureType classifyGesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const boost::circular_buffer<bool> &fastMotionFlags) const;

	const std::string getGestureName(const GestureType &type) const;

private:
	void createGesturePatternHistograms();

private:
	const double gesturePatternHistogramSigma_;
	const size_t binNum_;
	const double histogramNormalizationFactor_;

	std::vector<cv::MatND> gesturePatternHistograms_;
};

GestureClassifier::GestureClassifier(const size_t binNum, const double histogramNormalizationFactor)
: gesturePatternHistogramSigma_(1.0), binNum_(binNum), histogramNormalizationFactor_(histogramNormalizationFactor), gesturePatternHistograms_()
{
	createGesturePatternHistograms();
}

GestureClassifier::GestureType GestureClassifier::classifyGesture(const boost::circular_buffer<size_t> &matchedHistogramIndexes, const boost::circular_buffer<bool> &fastMotionFlags) const
{
	//const bool &isFull = matchedHistogramIndexes.full();
	//const size_t &count = matchedHistogramIndexes.size();
	//const size_t minCount = count / 2 + 1;

#if 0
	const int histDims = 1;
	const int phaseHistSize[] = { binNum_ };
	const float phaseHistRange1[] = { 0, binNum_ };
	const float *phaseHistRanges[] = { phaseHistRange1 };
	// we compute the histogram from the 0-th channel
	const int phaseHistChannels[] = { 0 };

	cv::MatND hist;
	cv::calcHist(
		&cv::Mat(std::vector<unsigned char>(matchedHistogramIndexes.begin(), matchedHistogramIndexes.end())),
		1, phaseHistChannels, cv::Mat(), hist, histDims, phaseHistSize, phaseHistRanges, true, false
	);
#else
	cv::MatND hist = cv::MatND::zeros(binNum_, 1, CV_32F);
	float *binPtr = (float *)hist.data;
	for (boost::circular_buffer<size_t>::const_iterator it = matchedHistogramIndexes.begin(); it != matchedHistogramIndexes.end(); ++it)
		if ((size_t)-1 != *it) ++(binPtr[*it]);
#endif

	// match histogram
	double minHistDist = std::numeric_limits<double>::max();
	const size_t &matchedIdx = HistogramMatcher::match(gesturePatternHistograms_, hist, minHistDist);

	// FIXME [delete] >>
	//std::cout << "\t\t\t*** " << minHistDist << std::endl;

	// TODO [adjust] >> design parameter
	const double minHistDistThreshold = 0.75;
	const size_t fastMotionCountThreshold = 1;
	// FIXME [modify] >>
	if (minHistDist < minHistDistThreshold)
	{
		switch (matchedIdx)
		{
		case 1:
			if ((size_t)std::count(fastMotionFlags.begin(), fastMotionFlags.end(), true) >= fastMotionCountThreshold)
				return GT_LEFT_FAST_MOVE;
			else
				return GT_LEFT_MOVE;
		case 2:
			if ((size_t)std::count(fastMotionFlags.begin(), fastMotionFlags.end(), true) >= fastMotionCountThreshold)
				return GT_RIGHT_FAST_MOVE;
			else
				return GT_RIGHT_MOVE;
		case 3:
			return GT_UP_MOVE;
		case 4:
			return GT_DOWN_MOVE;
		}
	}

	// FIXME [implement] >>
/*
	switch ()
	{
	case 5:
		return GT_HORIZONTAL_FLIP;
	case 6:
		return GT_VERTICAL_FLIP;
	case :
		return GT_JAMJAM;
	case :
		return GT_LEFT_90_TURN;
	case :
		return GT_RIGHT_90_TURN;
	case :
		return GT_CW;
	case :
		return CT_CCW;
	case :
		return GT_INFINITY;
	case :
		return GT_TRIANGLE;
	}
*/

	return GT_UNDEFINED;
}

const std::string GestureClassifier::getGestureName(const GestureType &type) const
{
	switch (type)
	{
	case GT_LEFT_MOVE:
		return "Left Move";
	case GT_RIGHT_MOVE:
		return "Right Move";
	case GT_UP_MOVE:
		return "Up Move";
	case GT_DOWN_MOVE:
		return "Down Move";
	case GT_LEFT_FAST_MOVE:
		return "Left Fast Move";
	case GT_RIGHT_FAST_MOVE:
		return "Right Fast Move";
	case GT_HORIZONTAL_FLIP:
		return "Horizontal Flip";
	case GT_VERTICAL_FLIP:
		return "Vertical Flip";
	case GT_JAMJAM:
		return "JamJam";
	case GT_LEFT_90_TURN:
		return "Left 90 Turn";
	case GT_RIGHT_90_TURN:
		return "Right 90 Turn";
	case GT_CW:
		return "CW";
	case CT_CCW:
		return "CCW";
	case GT_INFINITY:
		return "Infinity";
	case GT_TRIANGLE:
		return "Triangle";
	default:
		return "-----";
	}
}

void GestureClassifier::createGesturePatternHistograms()
{
	// create gesture pattern histograms
	GestureIdPatternHistogramGenerator gestureIdPatternHistogramGenerator(gesturePatternHistogramSigma_);
	gestureIdPatternHistogramGenerator.createHistograms(binNum_, histogramNormalizationFactor_);
	const std::vector<cv::MatND> &gestureIdPatternHistograms = gestureIdPatternHistogramGenerator.getHistograms();

	gesturePatternHistograms_.assign(gestureIdPatternHistograms.begin(), gestureIdPatternHistograms.end());

#if 0
	// FIXME [delete] >>
	// draw gesture pattern histograms
	const int indexHistBinWidth = 5, indexHistMaxHeight = 100;
	for (std::vector<cv::MatND>::const_iterator it = gestureIdPatternHistograms.begin(); it != gestureIdPatternHistograms.end(); ++it)
	{
#if 0
		double maxVal = 0.0;
		cv::minMaxLoc(*it, NULL, &maxVal, NULL, NULL);
#else
		const double maxVal = histogramNormalizationFactor;
#endif

		// draw 1-D histogram
		cv::Mat histImg(cv::Mat::zeros(indexHistMaxHeight, binNum_*indexHistBinWidth, CV_8UC3));
		my_opencv::draw_histogram_1D(*it, binNum_, maxVal, indexHistBinWidth, indexHistMaxHeight, histImg);

		cv::imshow(windowName4, histImg);
		cv::waitKey(0);
	}
#endif
}

void gesture_recognition_by_histogram(cv::VideoCapture &capture)
{
	const std::string windowName1("gesture recognition by histogram");
	const std::string windowName2("gesture recognition - actual histogram");
	const std::string windowName3("gesture recognition - matched histogram");
	const std::string windowName4("gesture recognition - matched id histogram");
	const std::string windowName5("gesture recognition - magnitude histogram");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName3, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName4, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName5, cv::WINDOW_AUTOSIZE);

	// TODO [adjust] >> design parameter
	const size_t ACCUMULATED_HISTOGRAM_NUM = 20;
	const size_t ACCUMULATED_HISTOGRAM_NUM_FOR_SHORT_TIME_GESTURE = 3;  // must be less thna and equal to ACCUMULATED_HISTOGRAM_NUM
	const size_t MAX_MATCHED_HISTOGRAM_NUM = 10;
	const double HISTOGRAM_NORMALIZATION_FACTOR = 5000.0;

	const bool does_apply_magnitude_filtering = true;
	const double magnitude_filtering_threshold_ratio = 0.3;
	const bool does_apply_magnitude_weighting = false;  // FIXME [implement] >> not yet supported

	float histogram_time_weight[ACCUMULATED_HISTOGRAM_NUM] = { 0.0f, };
	float weight_sum = 0.0f;
	for (size_t i = 0; i < ACCUMULATED_HISTOGRAM_NUM; ++i)
	{
		const float weight = std::exp(-(float)i / (float)ACCUMULATED_HISTOGRAM_NUM);
		histogram_time_weight[i] = weight;
		weight_sum += weight;
	}
	for (size_t i = 0; i < ACCUMULATED_HISTOGRAM_NUM; ++i)
		histogram_time_weight[i] /= weight_sum;

	const float fast_motion_threshold = 10.0;
	const float fast_motion_threshold_ratio = 0.3f;

	// TODO [check] >>
	//HistogramAccumulator histogramAccumulator(ACCUMULATED_HISTOGRAM_NUM, std::vector<float>(histogram_time_weight, histogram_time_weight + ACCUMULATED_HISTOGRAM_NUM));
	HistogramAccumulator histogramAccumulator(ACCUMULATED_HISTOGRAM_NUM);
	boost::circular_buffer<size_t> matched_histogram_indexes(MAX_MATCHED_HISTOGRAM_NUM);
	boost::circular_buffer<bool> fast_motion_flags(MAX_MATCHED_HISTOGRAM_NUM);

	// histograms' parameters
	const int histDims = 1;

	const int phaseHistBins = 360;
	const int phaseHistSize[] = { phaseHistBins };
	// phase varies from 0 to 359
	const float phaseHistRange1[] = { 0, phaseHistBins };
	const float *phaseHistRanges[] = { phaseHistRange1 };
	// we compute the histogram from the 0-th channel
	const int phaseHistChannels[] = { 0 };
	const int phaseHistBinWidth = 1, phaseHistMaxHeight = 100;

	const int magHistBins = 30;
	const int magHistSize[] = { magHistBins };
	// magnitude varies from 1 to 30
	const float magHistRange1[] = { 1, magHistBins + 1 };
	const float *magHistRanges[] = { magHistRange1 };
	// we compute the histogram from the 0-th channel
	const int magHistChannels[] = { 0 };
	const int magHistBinWidth = 5, magHistMaxHeight = 100;

	const int indexHistBins = ReferenceHistogramGenerator::REF_HISTOGRAM_NUM;
	const int indexHistSize[] = { indexHistBins };
	const float indexHistRange1[] = { 0, indexHistBins };
	const float *indexHistRanges[] = { indexHistRange1 };
	// we compute the histogram from the 0-th channel
	const int indexHistChannels[] = { 0 };
	const int indexHistBinWidth = 5, indexHistMaxHeight = 100;

	cv::MatND hist;

	// create reference histograms
	const double ref_histogram_sigma = 8.0;
	ReferenceHistogramGenerator refHistogramGenerator(ref_histogram_sigma);
	refHistogramGenerator.createHistograms(phaseHistBins, HISTOGRAM_NORMALIZATION_FACTOR);
	const std::vector<cv::MatND> &refHistograms = refHistogramGenerator.getHistograms();

#if 0
	// FIXME [delete] >>
	// draw reference histograms
	for (std::vector<cv::MatND>::const_iterator it = refHistograms.begin(); it != refHistograms.end(); ++it)
	{
#if 0
		double maxVal = 0.0;
		cv::minMaxLoc(*it, NULL, &maxVal, NULL, NULL);
#else
		const double maxVal = HISTOGRAM_NORMALIZATION_FACTOR * 0.05;
#endif

		// draw 1-D histogram
		cv::Mat histImg(cv::Mat::zeros(phaseHistMaxHeight, phaseHistBins*phaseHistBinWidth, CV_8UC3));
		my_opencv::draw_histogram_1D(*it, phaseHistBins, maxVal, phaseHistBinWidth, phaseHistMaxHeight, histImg);

		cv::imshow(windowName3, histImg);
		cv::waitKey(0);
	}
#endif

	// gesture classifier
	const GestureClassifier gestureClassifier(ReferenceHistogramGenerator::REF_HISTOGRAM_NUM, MAX_MATCHED_HISTOGRAM_NUM);

	const double MHI_TIME_DURATION = 1.0;

	cv::Mat prevgray, gray, frame, frame2;
	cv::Mat mhi, img, tmp_img;
	for (;;)
	{
		const double timestamp = (double)std::clock() / CLOCKS_PER_SEC;  // get current time in seconds

#if 1
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}
#else
		capture >> frame2;
		if (frame2.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}

		if (frame2.cols != imageWidth || frame2.rows != imageHeight)
		{
			//cv::resize(frame2, frame, cv::Size(imageWidth, imageHeight), 0.0, 0.0, cv::INTER_LINEAR);
			cv::pyrDown(frame2, frame);
		}
		else frame = frame2;
#endif

		cv::cvtColor(frame, gray, CV_BGR2GRAY);

		// smoothing
#if 0
		// METHOD #1: down-scale and up-scale the image to filter out the noise.

		{
			cv::Mat tmp;
			cv::pyrDown(gray, tmp);
			cv::pyrUp(tmp, gray);
		}
#elif 0
		// METHOD #2: Gaussian filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int kernelSize = 3;
			const double sigma = 2.0;
			cv::GaussianBlur(gray, gray, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);
		}
#elif 0
		// METHOD #3: box filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int ddepth = -1;  // the output image depth. -1 to use src.depth().
			const int kernelSize = 5;
			const bool normalize = true;
			cv::boxFilter(gray.clone(), gray, ddepth, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), normalize, cv::BORDER_DEFAULT);
			//cv::blur(gray.clone(), gray, cv::Size(kernelSize, kernelSize), cv::Point(-1, -1), cv::BORDER_DEFAULT);  // use the normalized box filter.
		}
#elif 0
		// METHOD #4: bilateral filtering.

		{
			// FIXME [adjust] >> adjust parameters.
			const int diameter = -1;  // diameter of each pixel neighborhood that is used during filtering. if it is non-positive, it is computed from sigmaSpace.
			const double sigmaColor = 3.0;  // for range filter.
			const double sigmaSpace = 50.0;  // for space filter.
			cv::bilateralFilter(gray.clone(), gray, diameter, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
		}
#else
		// METHOD #5: no filtering.

		//gray = gray;
#endif

		cv::cvtColor(gray, img, CV_GRAY2BGR);

		if (!prevgray.empty())
		{
			if (mhi.empty())
				mhi.create(gray.rows, gray.cols, CV_32FC1);

			cv::Mat processed_mhi, component_label_map;
			std::vector<cv::Rect> component_rects;
			my_opencv::segment_motion_using_mhi(timestamp, MHI_TIME_DURATION, prevgray, gray, mhi, processed_mhi, component_label_map, component_rects);

			//
			{
				double minVal = 0.0, maxVal = 0.0;
				cv::minMaxLoc(processed_mhi, &minVal, &maxVal);
				minVal = maxVal - 1.5 * MHI_TIME_DURATION;

				const double scale = (255.0 - 1.0) / (maxVal - minVal);
				const double offset = 1.0 - scale * minVal;
				processed_mhi.convertTo(tmp_img, CV_8UC1, scale, offset);

				// TODO [decide] >> want to use it ?
				tmp_img.setTo(cv::Scalar(0), component_label_map == 0);

				cv::cvtColor(tmp_img, img, CV_GRAY2BGR);
				img.setTo(cv::Scalar(255,0,0), processed_mhi >= (timestamp - 1.0e-20));  // last silhouette
			}

			// TODO [check] >> unexpected result
			// it happens that the component areas obtained by MHI disappear in motion, especially when changing motion direction
			if (component_rects.empty())
			{
				//std::cout << "************************************************" << std::endl;
				continue;
			}

			size_t k = 1;
			double min_dist = std::numeric_limits<double>::max();
			cv::Rect selected_rect;
			const double center_x = img.size().width * 0.5, center_y = img.size().height * 0.5;
			for (std::vector<cv::Rect>::const_iterator it = component_rects.begin(); it != component_rects.end(); ++it, ++k)
			{
				// reject very small components
				if (it->area() < 100 || it->width + it->height < 100)
					continue;

				// check for the case of little motion
				const size_t count = (size_t)cv::countNonZero((component_label_map == k)(*it));
				if (count < it->width * it->height * 0.05)
					continue;

				cv::rectangle(img, it->tl(), it->br(), CV_RGB(63, 0, 0), 2, 8, 0);

				const double x = it->x + it->width * 0.5, y = it->y + it->height * 0.5;
				const double dist = (x - center_x)*(x - center_x) + (y - center_y)*(y - center_y);
				if (dist < min_dist)
				{
					min_dist = dist;
					selected_rect = *it;
				}
			}

			if (selected_rect.area() > 0 &&
				(selected_rect.area() <= gray.rows * gray.cols / 2))  // reject too large area
				//selected_rect.area() <= 1.5 * average_area)  // reject too much area variation
			{
				cv::rectangle(img, selected_rect.tl(), selected_rect.br(), CV_RGB(255, 0, 0), 2, 8, 0);

				//
				cv::Mat flow;
				// FIXME [change] >> change parameters for large motion
				cv::calcOpticalFlowFarneback(prevgray(selected_rect), gray(selected_rect), flow, 0.25, 7, 15, 3, 7, 1.5, 0);

				std::vector<cv::Mat> flows;
				cv::split(flow, flows);

				cv::Mat flow_phase, flow_mag;
				cv::phase(flows[0], flows[1], flow_phase, true);  // return type: CV_32F
				cv::magnitude(flows[0], flows[1], flow_mag);  // return type: CV_32F

				// filter by magnitude
				if (does_apply_magnitude_filtering)
				{
					double minVal = 0.0, maxVal = 0.0;
					cv::minMaxLoc(flow_mag, &minVal, &maxVal, NULL, NULL);
					const double mag_threshold = minVal + (maxVal - minVal) * magnitude_filtering_threshold_ratio;

					// TODO [check] >> magic number, -1 is correct ?
					flow_phase.setTo(cv::Scalar::all(-1), flow_mag < mag_threshold);

					flow_mag.setTo(cv::Scalar::all(0), flow_mag < mag_threshold);
				}

				// calculate phase histogram
				cv::calcHist(&flow_phase, 1, phaseHistChannels, cv::Mat(), hist, histDims, phaseHistSize, phaseHistRanges, true, false);
				histogramAccumulator.addHistogram(hist);

				const int mag_count1 = cv::countNonZero(flow_mag > 0);
				if (mag_count1)
				{
					const int mag_count2 = cv::countNonZero(flow_mag > fast_motion_threshold);
					fast_motion_flags.push_back((float)mag_count2 / (float)mag_count1 > fast_motion_threshold_ratio);
				}
				else fast_motion_flags.push_back(false);

				// FIXME [delete] >> magnitude histogram
				{
					// calculate magnitude histogram
					cv::calcHist(&flow_mag, 1, magHistChannels, cv::Mat(), hist, histDims, magHistSize, magHistRanges, true, false);
					// normalize histogram
					my_opencv::normalize_histogram(hist, HISTOGRAM_NORMALIZATION_FACTOR);

					// draw magnitude histogram
					cv::Mat histImg(cv::Mat::zeros(magHistMaxHeight, magHistBins*magHistBinWidth, CV_8UC3));
					const double maxVal = HISTOGRAM_NORMALIZATION_FACTOR;
					my_opencv::draw_histogram_1D(hist, magHistBins, maxVal, magHistBinWidth, magHistMaxHeight, histImg);

					const int &fast_motion_threshold_pos = cvRound(fast_motion_threshold);
					cv::line(
						histImg, cv::Point((fast_motion_threshold_pos - 1) * magHistBinWidth, 0), cv::Point((fast_motion_threshold_pos - 1) * magHistBinWidth, magHistMaxHeight),
						CV_RGB(255, 0, 0), 1, 8, 0
					);

					cv::imshow(windowName5, histImg);
				}
			}
			else
			{
				histogramAccumulator.clearAllHistograms();
				matched_histogram_indexes.clear();
				fast_motion_flags.clear();
			}

#if 0
			if (histogramAccumulator.isFull())
#else
			if (histogramAccumulator.getHistogramSize() >= ACCUMULATED_HISTOGRAM_NUM_FOR_SHORT_TIME_GESTURE)
#endif
			{
				// accumulate phase histograms
				histogramAccumulator.accumulateHistograms();

				// normalize histogram
				cv::MatND &accumulated_hist = histogramAccumulator.getAccumulatedHistogram();
				my_opencv::normalize_histogram(accumulated_hist, HISTOGRAM_NORMALIZATION_FACTOR);

				//
#if 0
				double maxVal = 0.0;
				cv::minMaxLoc(accumulated_hist, NULL, &maxVal, NULL, NULL);
#else
				const double maxVal = HISTOGRAM_NORMALIZATION_FACTOR * 0.05;
#endif

				// draw accumulated phase histogram
				{
					cv::Mat histImg(cv::Mat::zeros(phaseHistMaxHeight, phaseHistBins*phaseHistBinWidth, CV_8UC3));
					my_opencv::draw_histogram_1D(accumulated_hist, phaseHistBins, maxVal, phaseHistBinWidth, phaseHistMaxHeight, histImg);

					cv::imshow(windowName2, histImg);
				}

				// match histogram
				double min_hist_dist = std::numeric_limits<double>::max();
				const size_t &matched_idx = HistogramMatcher::match(refHistograms, accumulated_hist, min_hist_dist);

				// TODO [adjust] >> design parameter
				const double hist_dist_threshold = 0.6;
				if (min_hist_dist < hist_dist_threshold)
				{
					matched_histogram_indexes.push_back(matched_idx);

					// classify gesture
					const GestureClassifier::GestureType &gesture_id = gestureClassifier.classifyGesture(matched_histogram_indexes, fast_motion_flags);

					// FIXME [modify]
					{
						cv::putText(img, gestureClassifier.getGestureName(gesture_id), cv::Point(10, 25), cv::FONT_HERSHEY_COMPLEX, 1.0, CV_RGB(255, 0, 0), 1, 8, false);
					}

					// FIXME [modify] >> delete ???
					{
						// calculate matched index histogram
						cv::MatND hist2;
#if defined(__GNUC__)
                        {
                            const cv::Mat matched_histogram_indexes_mat(std::vector<unsigned char>(matched_histogram_indexes.begin(), matched_histogram_indexes.end()));
                            cv::calcHist(&matched_histogram_indexes_mat, 1, indexHistChannels, cv::Mat(), hist2, histDims, indexHistSize, indexHistRanges, true, false);
                        }
#else
						cv::calcHist(&cv::Mat(std::vector<unsigned char>(matched_histogram_indexes.begin(), matched_histogram_indexes.end())), 1, indexHistChannels, cv::Mat(), hist2, histDims, indexHistSize, indexHistRanges, true, false);
#endif

						// normalize histogram
						//my_opencv::normalize_histogram(hist2, MAX_MATCHED_HISTOGRAM_NUM);

						// draw matched index histogram
						cv::Mat histImg2(cv::Mat::zeros(indexHistMaxHeight, indexHistBins*indexHistBinWidth, CV_8UC3));
						my_opencv::draw_histogram_1D(hist2, indexHistBins, MAX_MATCHED_HISTOGRAM_NUM, indexHistBinWidth, indexHistMaxHeight, histImg2);

						std::ostringstream sstream;
						sstream << "count: " << matched_histogram_indexes.size();;
						cv::putText(histImg2, sstream.str(), cv::Point(10, 15), cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(255, 0, 255), 1, 8, false);

						cv::imshow(windowName4, histImg2);
					}

					// FIXME [delete] >>
					std::cout << matched_idx << ", " << min_hist_dist << std::endl;

					// draw matched reference histogram
					{
						cv::Mat refHistImg(cv::Mat::zeros(phaseHistMaxHeight, phaseHistBins*phaseHistBinWidth, CV_8UC3));
						my_opencv::draw_histogram_1D(refHistograms[matched_idx], phaseHistBins, maxVal, phaseHistBinWidth, phaseHistMaxHeight, refHistImg);

						std::ostringstream sstream;
						sstream << "idx: " << matched_idx;
						cv::putText(refHistImg, sstream.str(), cv::Point(10, 15), cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(255, 0, 255), 1, 8, false);

						cv::imshow(windowName3, refHistImg);
					}
				}
				else
				{
					matched_histogram_indexes.push_back(-1);

					// FIXME [delete] >>
					std::cout << "-----------, " << min_hist_dist << std::endl;

					//cv::imshow(windowName3, cv::Mat::zeros(phaseHistMaxHeight, phaseHistBins*phaseHistBinWidth, CV_8UC3));
				}
			}

			if (!img.empty())
				cv::imshow(windowName1, img);
		}

		if (cv::waitKey(1) >= 0)
			break;

		std::swap(prevgray, gray);
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
	cv::destroyWindow(windowName3);
	cv::destroyWindow(windowName4);
	cv::destroyWindow(windowName5);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void gesture_recognition()
{
#if 1
	const int imageWidth = 640, imageHeight = 480;

	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

/*
	const double &propPosMsec = capture.get(CV_CAP_PROP_POS_MSEC);
	const double &propPosFrames = capture.get(CV_CAP_PROP_POS_FRAMES);
	const double &propPosAviRatio = capture.get(CV_CAP_PROP_POS_AVI_RATIO);
	const double &propFrameWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	const double &propFrameHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	const double &propFps = capture.get(CV_CAP_PROP_FPS);
	const double &propFourCC = capture.get(CV_CAP_PROP_FOURCC);
	const double &propFrameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);
	const double &propFormat = capture.get(CV_CAP_PROP_FORMAT);
	const double &propMode = capture.get(CV_CAP_PROP_MODE);
	const double &propBrightness = capture.get(CV_CAP_PROP_BRIGHTNESS);
	const double &propContrast = capture.get(CV_CAP_PROP_CONTRAST);
	const double &propSaturation = capture.get(CV_CAP_PROP_SATURATION);
	const double &propHue = capture.get(CV_CAP_PROP_HUE);
	const double &propGain = capture.get(CV_CAP_PROP_GAIN);
	const double &propExposure = capture.get(CV_CAP_PROP_EXPOSURE);
	const double &propConvertRGB = capture.get(CV_CAP_PROP_CONVERT_RGB);
	const double &propWhiteBalance = capture.get(CV_CAP_PROP_WHITE_BALANCE);
	const double &propRectification = capture.get(CV_CAP_PROP_RECTIFICATION);
	const double &propMonochrome = capture.get(CV_CAP_PROP_MONOCROME);

	capture.set(CV_CAP_PROP_POS_MSEC, propPosMsec);
	capture.set(CV_CAP_PROP_POS_FRAMES, propPosFrames);
	capture.set(CV_CAP_PROP_POS_AVI_RATIO, propPosAviRatio);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, propFrameWidth);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, propFrameHeight);
	capture.set(CV_CAP_PROP_FPS, propFps);
	capture.set(CV_CAP_PROP_FOURCC, propFourCC);
	capture.set(CV_CAP_PROP_FRAME_COUNT, propFrameCount);
	capture.set(CV_CAP_PROP_FORMAT, propFormat);
	capture.set(CV_CAP_PROP_MODE, propMode);
	capture.set(CV_CAP_PROP_BRIGHTNESS, propBrightness);
	capture.set(CV_CAP_PROP_CONTRAST, propContrast);
	capture.set(CV_CAP_PROP_SATURATION, propSaturation);
	capture.set(CV_CAP_PROP_HUE, propHue);
	capture.set(CV_CAP_PROP_GAIN, propGain);
	capture.set(CV_CAP_PROP_EXPOSURE, propExposure);
	capture.set(CV_CAP_PROP_CONVERT_RGB, propConvertRGB);
	capture.set(CV_CAP_PROP_WHITE_BALANCE, propWhiteBalance);
	capture.set(CV_CAP_PROP_RECTIFICATION, propRectification);
	capture.set(CV_CAP_PROP_MONOCROME, propMonochrome);
*/
	capture.set(CV_CAP_PROP_FRAME_WIDTH, imageWidth);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, imageHeight);
#else
	//const std::string avi_filename("./data/machine_vision/opencv/flycap-0001.avi");
	//const std::string avi_filename("./data/machine_vision/opencv/tree.avi");
	const std::string avi_filename("./data/machine_vision/opencv/s01_g01_1_ccw_normal.avi");

	//const int imageWidth = 640, imageHeight = 480;

	cv::VideoCapture capture(avi_filename);
	if (!capture.isOpened())
	{
		std::cout << "a video file not found" << std::endl;
		return;
	}
#endif

	//local::gesture_recognition_by_frequency(capture);
	local::gesture_recognition_by_histogram(capture);
}

}  // namespace my_opencv
