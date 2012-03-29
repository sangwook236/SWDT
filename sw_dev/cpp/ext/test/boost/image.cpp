#if 0
#include <boost/gil/gil_all.hpp>
#else
#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/color_convert.hpp>
#include <boost/gil/extension/dynamic_image/any_image.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>
#include <boost/gil/extension/io/jpeg_dynamic_io.hpp>
#include <boost/gil/extension/numeric/sampler.hpp>
#include <boost/gil/extension/numeric/resample.hpp>
#include <boost/gil/extension/numeric/kernel.hpp>
#include <boost/gil/extension/numeric/convolve.hpp>
#endif
#include <boost/mpl/vector.hpp>
#include <iostream>


namespace {
namespace local {

void resize()
{
	boost::gil::rgb8_image_t img;
	boost::gil::jpeg_read_image("./boost_data/image/lena_rgb.jpg", img);

	// scale the image to 100x100 pixels using bilinear resampling
	boost::gil::rgb8_image_t square100x100(100, 100);
	boost::gil::resize_view(boost::gil::const_view(img), boost::gil::view(square100x100), boost::gil::bilinear_sampler());
	boost::gil::jpeg_write_view("./boost_data/image/lena_rgb.resize_out.jpg", boost::gil::const_view(square100x100));
}

void dynamic_image()
{
	typedef boost::mpl::vector<boost::gil::gray8_image_t, boost::gil::rgb8_image_t, boost::gil::gray16_image_t, boost::gil::rgb16_image_t> my_images_t;

	boost::gil::any_image<my_images_t> dynamic_img;
	boost::gil::jpeg_read_image("./boost_data/image/lena_rgb.jpg", dynamic_img);

	// save the image upside down, preserving its native color space and channel depth
	boost::gil::jpeg_write_view("./boost_data/image/lena_rgb.dynamic_image_out.jpg", boost::gil::flipped_up_down_view(boost::gil::const_view(dynamic_img)));
}

void affine()
{
	boost::gil::rgb8_image_t img;
	boost::gil::jpeg_read_image("./boost_data/image/lena_rgb.jpg",img);

	// transform the image by an arbitrary affine transformation using nearest-neighbor resampling
	boost::gil::rgb8_image_t transf(boost::gil::rgb8_image_t::point_t(boost::gil::view(img).dimensions() * 2));
	boost::gil::fill_pixels(boost::gil::view(transf), boost::gil::rgb8_pixel_t(255, 0, 0));  // the background is red

	const boost::gil::matrix3x2<double> mat = boost::gil::matrix3x2<double>::get_translate(-boost::gil::point2<double>(200, 250)) * boost::gil::matrix3x2<double>::get_rotate(-15 * 3.14 / 180.0);
	boost::gil::resample_pixels(boost::gil::const_view(img), boost::gil::view(transf), mat, boost::gil::nearest_neighbor_sampler());
	boost::gil::jpeg_write_view("./boost_data/image/lena_rgb.affine_out.jpg", boost::gil::view(transf));
}

template <typename GrayView, typename R>
void gray_image_hist(const GrayView &img_view, R &hist)
{
	//for_each_pixel(img_view, ++boost::lambda::var(hist)[boost::lambda::_1]);
	for (typename GrayView::iterator it = img_view.begin(); it != img_view.end(); ++it)
		++hist[*it];
}

template <typename V, typename R>
void get_hist(const V &img_view, R &hist)
{
    gray_image_hist(boost::gil::color_converted_view<boost::gil::gray8_pixel_t>(img_view), hist);
}

void histogram()
{
	boost::gil::rgb8_image_t img;
	boost::gil::jpeg_read_image("./boost_data/image/lena_rgb.jpg", img);

	int histogram[256];
	std::fill(histogram, histogram + 256, 0);
	get_hist(boost::gil::const_view(img), histogram);

	for (std::size_t ii = 0; ii < 256; ++ii)
		std::cout << histogram[ii] << ", ";
	std::cout << std::endl;
}

template <typename Out>
struct halfdiff_cast_channels
{
    template <typename T>
	Out operator()(const T &in1, const T &in2) const
	{
        return Out((in2 - in1) / 2);
    }
};

template <typename SrcView, typename DstView>
void x_gradient(const SrcView &src, const DstView &dst)
{
	typedef typename boost::gil::channel_type<DstView>::type dst_channel_t;

	for (int y = 0; y < src.height(); ++y)
	{
		typename SrcView::x_iterator src_it = src.row_begin(y);
		typename DstView::x_iterator dst_it = dst.row_begin(y);

		for (int x = 1; x < src.width() - 1; ++x)
			boost::gil::static_transform(src_it[x-1], src_it[x+1], dst_it[x], halfdiff_cast_channels<dst_channel_t>());
	}
}

void gradient()
{
	boost::gil::rgb8_image_t img;
	boost::gil::jpeg_read_image("./boost_data/image/lena_rgb.jpg", img);

	boost::gil::gray8s_image_t img_out(img.dimensions());
	boost::gil::fill_pixels(boost::gil::view(img_out), boost::gil::bits8s(0));

	typedef boost::gil::pixel<typename boost::gil::channel_type<const boost::gil::rgb8_image_t::const_view_t>::type, boost::gil::gray_layout_t> gray_pixel_t;
	x_gradient(boost::gil::color_converted_view<gray_pixel_t>(boost::gil::const_view(img)), boost::gil::view(img_out));
	boost::gil::jpeg_write_view("./boost_data/image/lena_rgb.x_gradient_out.jpg", boost::gil::color_converted_view<boost::gil::gray8_pixel_t>(boost::gil::const_view(img_out)));
}

void convolution()
{
	boost::gil::rgb8_image_t img;
	boost::gil::jpeg_read_image("./boost_data/image/lena_rgb.jpg", img);

	// convolve the rows and the columns of the image with a fixed kernel
	boost::gil::rgb8_image_t convolved(img);

	// radius-1 Gaussian kernel, size 9
	const float gaussian_1[] = {
		0.00022923296f, 0.0059770769f, 0.060597949f,
		0.24173197f, 0.38292751f, 0.24173197f,
		0.060597949f, 0.0059770769f, 0.00022923296f
	};
/*
	// radius-2 Gaussian kernel, size 15
	const float gaussian_2[] = {
		0.00048869418f, 0.0024031631f, 0.0092463447f,
		0.027839607f, 0.065602221f, 0.12099898f, 0.17469721f,
		0.19744757f,
		0.17469721f, 0.12099898f, 0.065602221f, 0.027839607f,
		0.0092463447f, 0.0024031631f, 0.00048869418f
	};
	// radius-3 Gaussian kernel, size 23
	const float gaussian_3[] = {
		0.00016944126f, 0.00053842377f, 0.0015324751f, 0.0039068931f,
		0.0089216027f, 0.018248675f, 0.033434924f, 0.054872241f,
		0.080666073f, 0.10622258f, 0.12529446f,
		0.13238440f,
		0.12529446f, 0.10622258f, 0.080666073f,
		0.054872241f, 0.033434924f, 0.018248675f, 0.0089216027f,
		0.0039068931f, 0.0015324751f, 0.00053842377f, 0.00016944126f
	};
	// radius-4 Gaussian kernel, size 29
	const float gaussian_4[] = {
		0.00022466264f, 0.00052009715f, 0.0011314391f, 0.0023129794f,
		0.0044433107f, 0.0080211498f, 0.013606987f, 0.021691186f,
		0.032493830f, 0.045742013f, 0.060509924f, 0.075220309f,
		0.087870099f, 0.096459411f, 0.099505201f, 0.096459411f, 0.087870099f,
		0.075220309f, 0.060509924f, 0.045742013f, 0.032493830f,
		0.021691186f, 0.013606987f, 0.0080211498f, 0.0044433107f,
		0.0023129794f, 0.0011314391f, 0.00052009715f, 0.00022466264f,
	};
*/
	std::cout << "processing convolution ..." << std::endl;

	const boost::gil::kernel_1d_fixed<float, 9> kernel(gaussian_1, 4);

	boost::gil::convolve_rows_fixed<boost::gil::rgb32f_pixel_t>(boost::gil::const_view(convolved), kernel, boost::gil::view(convolved));
	boost::gil::convolve_cols_fixed<boost::gil::rgb32f_pixel_t>(boost::gil::const_view(convolved), kernel, boost::gil::view(convolved));
	boost::gil::jpeg_write_view("./boost_data/image/lena_rgb.convolution1_out.jpg", boost::gil::view(convolved));

	// use a resizable kernel
	const boost::gil::kernel_1d<float> kernel2(gaussian_1, 9 ,4);
	
	boost::gil::convolve_rows<boost::gil::rgb32f_pixel_t>(boost::gil::const_view(img), kernel2, boost::gil::view(img));
	boost::gil::convolve_cols<boost::gil::rgb32f_pixel_t>(boost::gil::const_view(img), kernel2, boost::gil::view(img));
	boost::gil::jpeg_write_view("./boost_data/image/lena_rgb.convolution2_out.jpg", boost::gil::view(img));
}

}  // local
}  // unnamed namespace

void image()
{
	local::resize();
	local::dynamic_image();

	local::affine();

	local::histogram();
	local::gradient();

	local::convolution();
}
