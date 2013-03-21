#ifndef FSTTHREADPOOL_H
#define FSTTHREADPOOL_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    thread_pool.hpp
   \brief   Implements thread scheduler that assigns jobs up to maximum number of threads allowed
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at http://fst.utia.cz
   \date    March 2011
   \version 3.1.0.beta
   \note    FST3 was developed using gcc 4.3 and requires
   \note    \li Boost library (http://www.boost.org/, tested with versions 1.33.1 and 1.44),
   \note    \li (\e optionally) LibSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/, 
                tested with version 3.00)
   \note    Note that LibSVM is required for SVM related tools only,
            as demonstrated in demo12t.cpp, demo23.cpp, demo25t.cpp, demo32t.cpp, etc.

*/ /* 
=========================================================================
Copyright:
  * FST3 software (with exception of any externally linked libraries) 
    is copyrighted by Institute of Information Theory and Automation (UTIA), 
    Academy of Sciences of the Czech Republic.
  * FST3 source codes as presented here do not contain code of third parties. 
    FST3 may need linkage to external libraries to exploit its functionality
    in full. For details on obtaining and possible usage restrictions 
    of external libraries follow their original sources (referenced from
    FST3 documentation wherever applicable).
  * FST3 software is available free of charge for non-commercial use. 
    Please address all inquires concerning possible commercial use 
    of FST3, or if in doubt, to FST3 maintainer (see http://fst.utia.cz).
  * Derivative works based on FST3 are permitted as long as they remain
    non-commercial only.
  * Re-distribution of FST3 software is not allowed without explicit
    consent of the copyright holder.
Disclaimer of Warranty:
  * FST3 software is presented "as is", without warranty of any kind, 
    either expressed or implied, including, but not limited to, the implied 
    warranties of merchantability and fitness for a particular purpose. 
    The entire risk as to the quality and performance of the program 
    is with you. Should the program prove defective, you assume the cost 
    of all necessary servicing, repair or correction.
Limitation of Liability:
  * The copyright holder will in no event be liable to you for damages, 
    including any general, special, incidental or consequential damages 
    arising out of the use or inability to use the code (including but not 
    limited to loss of data or data being rendered inaccurate or losses 
    sustained by you or third parties or a failure of the program to operate 
    with any other programs).
========================================================================== */

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/function.hpp>
#include <boost/static_assert.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include "global.hpp"

namespace FST {

//! Implements thread scheduler that assigns jobs up to maximum number of threads allowed
template<unsigned int max_threads=2>
class ThreadPool
{
public:
	ThreadPool() : _running_threads(0) 
	{
		BOOST_STATIC_ASSERT(max_threads>0); 
		for(unsigned int i=0;i<max_threads;i++) threads[i].tid=-1;
	}
	~ThreadPool() {notify("ThreadPool destructor, idx=",idx); join_all();} //{join_rest_cond();}
	
	unsigned int go(const boost::function0<void> &threadfunc) 
	{
		boost::mutex::scoped_lock lock_launcher(mutex_launcher);
		if(_running_threads==max_threads)
		{
			notify("Max allowed threads running, waiting for one to end...");
			while (_running_threads==max_threads) cond_launcher.wait(lock_launcher);
		}
		unsigned int free_tidx=0; 
		while(threads[free_tidx].tid!=-1 && free_tidx<max_threads) free_tidx++;
		assert(free_tidx<max_threads);
		{
			notify("Launching thread, tid=",idx,"...");
			{
				_running_threads++;
			}
			if(threads[free_tidx].m_thread) 
			{
				notify("Joining past ",free_tidx,"th thread before lanching new one.");
				threads[free_tidx].m_thread->join();
			}
			{
				threads[free_tidx].tid=idx++;
				threads[free_tidx].m_thread.reset(new boost::thread(boost::bind(&ThreadPool::launch_thread,this,threadfunc,free_tidx)));
				notify("Thread tid=",threads[free_tidx].tid," (",free_tidx,"th) launched...");
			}
		}
		return free_tidx;
	}
	
	void join_all() 
	{
		notify("Calling join_all...");
		for(unsigned int i=0; i<max_threads; i++) 
		{
			assert((threads[i].m_thread && threads[i].tid>=0) || (threads[i].tid==-1));
			if(threads[i].m_thread) 
			{
				notify("join_all: Joining ",i,"th thread.");
				threads[i].m_thread->join();
				threads[i].m_thread.reset();
				threads[i].tid=-1;
			}
		}
		_running_threads=0;
		notify("join_all finished...");
	}
	
	unsigned int get_running_threads() const {return _running_threads;}
	
private:
	// the next procedure must be re-entrant
	void launch_thread(const boost::function0<void> &threadfunc, unsigned int threadsid)
	{
		notify("Starting launch_thread(tid=",threads[threadsid].tid,", ",threadsid,"th) ...");
		threadfunc();
		notify("Finishing launch_thread(tid=",threads[threadsid].tid,", ",threadsid,"th).");
		{
			boost::mutex::scoped_lock lock(mutex_launcher);
			_running_threads--;
			threads[threadsid].tid=-1;
		}
		{
			cond_launcher.notify_one();
		}
	}
private:
	typedef boost::scoped_ptr<boost::thread> ThreadPtr;
	//! Structure to keep status of threads in the pool
	typedef struct {ThreadPtr m_thread; int tid;} ThreadsArray;
	ThreadsArray threads[max_threads];
	
private:
	static unsigned int idx;
	unsigned int _running_threads;

	boost::mutex mutex_launcher;
	boost::condition cond_launcher;
};

template<unsigned int max_threads>
unsigned int ThreadPool<max_threads>::idx=0;

} // namespace
#endif // FSTTHREADPOOL_H ///:~
