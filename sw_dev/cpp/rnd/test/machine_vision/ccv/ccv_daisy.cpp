#ifdef __cplusplus
extern "C" {
#endif
#include <ccv/ccv.h>
#ifdef __cplusplus
}
#endif
#include <boost/timer/timer.hpp>
#include <iostream>
#include <string>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_ccv {

void daisy()
{
	const std::string image_filename("./data/machine_vision/ccv/box.png");
	//const std::string image_filename("./data/machine_vision/ccv/book.png");

	ccv_enable_default_cache();

	ccv_dense_matrix_t *image = NULL;
#if defined(WIN32) || defined(_WIN32)
	ccv_read_impl(image_filename.c_str(), &image, CCV_IO_GRAY | CCV_IO_ANY_FILE, 0, 0, 0);
#else
	ccv_read(image_filename.c_str(), &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
#endif
    if (NULL == image)
    {
        std::cout << "an image file not found: " << image_filename << std::endl;
        return;
    }

	ccv_dense_matrix_t *a = ccv_dense_matrix_new(image->rows, image->cols, CCV_8U | CCV_C1, NULL, 0);
	for (int i = 0; i < image->rows; ++i)
		for (int j = 0; j < image->cols; ++j)
			//a->data.ptr[i * a->step + j] = (image->data.ptr[i * image->step + j * 3] * 29 + image->data.ptr[i * image->step + j * 3 + 1] * 61 + image->data.ptr[i * image->step + j * 3 + 2] * 10) / 100;
			a->data.u8[i * a->step + j] = (image->data.u8[i * image->step + j * 3] * 29 + image->data.u8[i * image->step + j * 3 + 1] * 61 + image->data.u8[i * image->step + j * 3 + 2] * 10) / 100;

	//
	ccv_daisy_param_t param;
	param.radius = 15;
	param.rad_q_no = 3;
	param.th_q_no = 8;
	param.hist_th_q_no = 8;
	param.normalize_threshold = 0.154f;
	param.normalize_method = CCV_DAISY_NORMAL_PARTIAL;

	//
	std::cout << "start processing ..." << std::endl;

	ccv_dense_matrix_t *x = NULL;
	{
		boost::timer::cpu_timer timer;

		ccv_daisy(a, &x, 0, param);

		const boost::timer::cpu_times elapsed_times(timer.elapsed());
		std::cout << "elpased time : " << (elapsed_times.system + elapsed_times.user) << " sec" << std::endl;
	}

	std::cout << "end processing ..." << std::endl;

	//
	int row = 0;
	int col = 0;
	//float *x_ptr = x->data.fl + row * x->cols + col * 200;
	float *x_ptr = x->data.f32 + row * x->cols + col * 200;
	for (int k = 0; k < 25; ++k)
	{
		for (int t = 0; t < 8; ++t)
			std::cout << x_ptr[k * 8 + t] << ' ';
		std::cout << std::endl;
	}

	ccv_matrix_free(image);
	ccv_matrix_free(a);
	ccv_matrix_free(x);

	//ccv_garbage_collect();
	ccv_disable_cache();
}

}  // namespace my_ccv
