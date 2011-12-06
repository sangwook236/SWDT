#include <vld/vld.h>
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


// vld configuration file:
//	vld.ini

struct thread_proc
{
	void operator()()
	{
		int *pi = new int [50];
		long *pl = new long [10];
		float *pf = new float [30];
		double *pd = new double [50];

		delete [] pi;
		pi = NULL;
		//delete [] pl;
		//pl = NULL;
		//delete [] pf;
		//pf = NULL;
		//delete [] pd;
		//pd = NULL;
	}
};

int main()
{
	boost::scoped_ptr<boost::thread> thrd(new boost::thread(thread_proc()));  // create thread

	{
		int *pi = new int [50];
		long *pl = new long [10];
		float *pf = new float [30];
		double *pd = new double [50];

		delete [] pi;
		pi = NULL;
		//delete [] pl;
		//pl = NULL;
		//delete [] pf;
		//pf = NULL;
		//delete [] pd;
		//pd = NULL;
	}

	thrd->join();

	std::cout << "press any key to terminate ..." << std::endl;
	std::cin.get();

	return 0;
}
