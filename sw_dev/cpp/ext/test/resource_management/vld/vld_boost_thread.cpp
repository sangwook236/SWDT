//#include <vld/vld.h>  // don't need to include here. the header file is included in main.cpp
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


namespace {
namespace local {

struct thread_proc
{
	thread_proc(const bool leakage)
	: leakage_(leakage)
	{}

	void operator()()
	{
		int *pi = new int [50];
		long *pl = new long [10];
		float *pf = new float [30];
		double *pd = new double [50];

		delete [] pi;
		pi = NULL;
		if (!leakage_)
		{
			delete [] pl;
			pl = NULL;
			delete [] pf;
			pf = NULL;
			delete [] pd;
			pd = NULL;
		}
	}

private:
	const bool leakage_;
};

}  // namespace local
}  // unnamed namespace

namespace vld {

void boost_thread(const bool leakage)
{
	boost::scoped_ptr<boost::thread> thrd(new boost::thread(local::thread_proc(leakage)));  // create thread

	{
		int *pi = new int [50];
		long *pl = new long [10];
		float *pf = new float [30];
		double *pd = new double [50];

		delete [] pi;
		pi = NULL;
		if (!leakage)
		{
			delete [] pl;
			pl = NULL;
			delete [] pf;
			pf = NULL;
			delete [] pd;
			pd = NULL;
		}
	}

	thrd->join();
}

}  // namespace vld
