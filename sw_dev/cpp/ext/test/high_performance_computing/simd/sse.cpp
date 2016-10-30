#include <boost/date_time/posix_time/posix_time.hpp>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <ctime>
#include <cstdlib>
#if defined(_WIN64) || defined(_WIN32)
#include <xmmintrin.h>
#include <windows.h>
#endif


#if defined(max)
#undef max
#endif
#if defined(min)
#undef min
#endif


namespace {
namespace local {

#if defined(_WIN64) || defined(_WIN32)
class Timer
{
public:
	Timer()
	{
		QueryPerformanceFrequency(&freq_);
		QueryPerformanceCounter(&startTime_);
	}
	~Timer()
	{
	}

	__int64 getElapsedTime()
	{
		LARGE_INTEGER endTime;
		QueryPerformanceCounter(&endTime);
		return (endTime.QuadPart - startTime_.QuadPart) * 1000 / freq_.QuadPart;
	}

private:
	LARGE_INTEGER freq_;
	LARGE_INTEGER startTime_;
};
#endif

void test1()
{
	const int array_size = 60000;

	__declspec(align(16)) float arr1[array_size]= { 0.0f, };
	__declspec(align(16)) float arr2[array_size] = { 0.0f, };
	__declspec(align(16)) float result[array_size] = { 0.0f, };
/*
	__declspec(align(16)) float *arr1 = (float *)_aligned_malloc(array_size * sizeof(float), 16);
	__declspec(align(16)) float *arr2 = (float *)_aligned_malloc(array_size * sizeof(float), 16);
	__declspec(align(16)) float *result = (float *)_aligned_malloc(array_size * sizeof(float), 16);
*/

	for (int i = 0; i < array_size; ++i)
	{
		arr1[i] = (float)i;
		arr2[i] = (float)(i - 30000);
	}

	{
		float *src1 = arr1;
		float *src2 = arr2;
		float *dest = result;

		//
		const boost::posix_time::ptime stime = boost::posix_time::microsec_clock::universal_time();
#if defined(_WIN64) || defined(_WIN32)
		Timer timer;
#endif

		for (int i = 0; i < array_size; ++i)
		{
			*dest = (float)std::sqrt((*src1) * (*src1) + (*src2) * (*src2)) + 0.5f;

			++src1;
			++src2;
			++dest;
		}

		const boost::posix_time::ptime etime = boost::posix_time::microsec_clock::universal_time();
		const boost::posix_time::time_duration td = etime - stime;

		std::cout << stime << " : " << etime << " : " << td << std::endl;
#if defined(_WIN64) || defined(_WIN32)
		std::cout << timer.getElapsedTime() << std::endl;
#endif
	}

	{
		const int loop_count = array_size / 4;

		__m128 m1, m2, m3, m4;

		__m128 *src1 = (__m128 *)arr1;
		__m128 *src2 = (__m128 *)arr2;
		__m128 *dest = (__m128 *)result;

		__m128 m0_5 = _mm_set_ps1(0.5f);  // m0_5[0, 1, 2, 3] = 0.5

		//
		const boost::posix_time::ptime stime = boost::posix_time::microsec_clock::universal_time();
#if defined(_WIN64) || defined(_WIN32)
		Timer timer;
#endif

		for (int i = 0; i < loop_count; ++i)
		{
			m1 = _mm_mul_ps(*src1, *src1);        // m1 = *src1 * *src1
			m2 = _mm_mul_ps(*src2, *src2);        // m2 = *src2 * *src2
			m3 = _mm_add_ps(m1, m2);              // m3 = m1 + m2
			m4 = _mm_sqrt_ps(m3);                 // m4 = sqrt(m3)
			*dest = _mm_add_ps(m4, m0_5);         // *dest = m4 + 0.5

			++src1;
			++src2;
			++dest;
		}

		const boost::posix_time::ptime etime = boost::posix_time::microsec_clock::universal_time();
		const boost::posix_time::time_duration td = etime - stime;

		std::cout << stime << " : " << etime << " : " << td << std::endl;
#if defined(_WIN64) || defined(_WIN32)
		std::cout << timer.getElapsedTime() << std::endl;
#endif
	}
/*
	_aligned_free(arr1);
	_aligned_free(arr2);
	_aligned_free(result);
*/
}

void test2()
{
	const int array_size = 60000;

	std::srand((unsigned int)time(NULL));

	__declspec(align(16)) float arr[array_size]= { 0.0f, };
	__declspec(align(16)) float result[array_size] = { 0.0f, };
/*
	__declspec(align(16)) float *arr = (float *)_aligned_malloc(array_size * sizeof(float), 16);
	__declspec(align(16)) float *result = (float *)_aligned_malloc(array_size * sizeof(float), 16);

*/

	for (int i = 0; i < array_size; ++i)
		arr[i] = (float)std::fabs((float)std::rand());

	{
		float fmin = FLT_MAX;
		float fmax = FLT_MIN;

		//
		const boost::posix_time::ptime stime = boost::posix_time::microsec_clock::universal_time();
#if defined(_WIN64) || defined(_WIN32)
		Timer timer;
#endif

		for (int i = 0; i < array_size; ++i)
		{
			result[i] = std::sqrt(arr[i] * 2.8f);

			if (result[i] < fmin)
				fmin = result[i];
			if (result[i] > fmax)
				fmax = result[i];
		}

		const boost::posix_time::ptime etime = boost::posix_time::microsec_clock::universal_time();
		const boost::posix_time::time_duration td = etime - stime;

		std::cout << stime << " : " << etime << " : " << td << std::endl;
#if defined(_WIN64) || defined(_WIN32)
		std::cout << timer.getElapsedTime() << std::endl;
#endif
	}

	{
		const int loop_count = array_size / 4;

		__m128 coeff = _mm_set_ps1(2.8f);  // coeff[0, 1, 2, 3] = 2.8
		__m128 tmp;

		__m128 min128 = _mm_set_ps1(FLT_MAX);  // min128[0, 1, 2, 3] = FLT_MAX
		__m128 max128 = _mm_set_ps1(FLT_MIN);  // max128[0, 1, 2, 3] = FLT_MIN

		__m128 *src = (__m128 *)arr;
		__m128 *dst = (__m128 *)result;

		//
		const boost::posix_time::ptime stime = boost::posix_time::microsec_clock::universal_time();
#if defined(_WIN64) || defined(_WIN32)
		Timer timer;
#endif

		for (int i = 0; i < loop_count; ++i)
		{
			tmp = _mm_mul_ps(*src, coeff);      // tmp = *src * coeff
			*dst = _mm_sqrt_ps(tmp);            // *dst = sqrt(tmp)

			min128 =  _mm_min_ps(*dst, min128);
			max128 =  _mm_max_ps(*dst, max128);

			++src;
			++dst;
		}

		// Extract minimum and maximum values from min128 and max128.
		union u
		{
			__m128 m;
			float f[4];
		} x;

		x.m = min128;
		//const float fmin = std::min(x.f[0], std::min(x.f[1], std::min(x.f[2], x.f[3])));
		const float fmin = *std::min_element(x.f, x.f + 4);

		x.m = max128;
		//const float fmax = std::max(x.f[0], std::max(x.f[1], std::max(x.f[2], x.f[3])));
		const float fmax = *std::max_element(x.f, x.f + 4);

		const boost::posix_time::ptime etime = boost::posix_time::microsec_clock::universal_time();
		const boost::posix_time::time_duration td = etime - stime;

		std::cout << stime << " : " << etime << " : " << td << std::endl;
#if defined(_WIN64) || defined(_WIN32)
		std::cout << timer.getElapsedTime() << std::endl;
#endif
	}
/*
	_aligned_free(arr);
	_aligned_free(result);
*/
}

}  // namespace local
}  // unnamed namespace

namespace my_simd {

void sse()
{
	local::test1();
	local::test2();
}

}  // namespace my_simd
