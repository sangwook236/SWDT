// Steven Farago
// Drew Lamar
//
// Numerical Analysis -- Linear Algebra
// Final Project
//
//
// Defines the TimerUtil class.  Used in the main code to determine the  
// runtime of the parser.
//

#ifndef _TIMERUTIL_H
#define _TIMERUTIL_H

#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <iostream>

using namespace std;

class TimerUtil {
 private:
  timeval startTime;  
  timeval stopTime;

  long sec;
  long microsec;
  timeval usrStartTime, sysStartTime;
 public:

  // Note: constructor automatically sets start time
  TimerUtil() {
    setStartTime();
  }

  // Call this at the time you wish to start timing.
  void setStartTime() {
    sec = microsec = 0;
    gettimeofday(&startTime, NULL);
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    usrStartTime.tv_sec = r_usage.ru_utime.tv_sec;
    usrStartTime.tv_usec = r_usage.ru_utime.tv_usec;
    sysStartTime.tv_sec = r_usage.ru_stime.tv_sec;
    sysStartTime.tv_usec = r_usage.ru_stime.tv_usec;
  }
  
  // Call this at the time you wish to end timing.
  void setStopTime() {
    gettimeofday(&stopTime, NULL);
    
    sec = stopTime.tv_sec - startTime.tv_sec;
    microsec = stopTime.tv_usec - startTime.tv_usec;
    
    if (microsec < 0) {
      microsec+=1000000;
      sec--;
    }
  }  
  
  void setStopTime(ostream& os, const char* notes) {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    
    long usec = r_usage.ru_utime.tv_sec - usrStartTime.tv_sec;
    long umsec = r_usage.ru_utime.tv_usec - usrStartTime.tv_usec;
    if (umsec < 0) {
      umsec += 1000000;
      usec--;
    }
    long ssec = r_usage.ru_stime.tv_sec - sysStartTime.tv_sec;
    long smsec = r_usage.ru_stime.tv_usec - sysStartTime.tv_usec;
    if (smsec < 0) {
      smsec += 1000000;
      ssec--;
    }

    setStopTime();
    os << notes << endl;
    os << "CPU Usage: user = " << usec
       << " seconds " << umsec << " ms, system = " << ssec 
       << " seconds " << smsec << "ms" << endl;
    os << "ELAPSED Time: " << sec << " seconds "
       << microsec << " ms" << endl;
  }

void setStopTime(ostream& os, const char* notes, const int iter) {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    
    long usec = r_usage.ru_utime.tv_sec - usrStartTime.tv_sec;
    long umsec = r_usage.ru_utime.tv_usec - usrStartTime.tv_usec;
    if (umsec < 0) {
      umsec += 1000000;
      usec--;
    }
    long ssec = r_usage.ru_stime.tv_sec - sysStartTime.tv_sec;
    long smsec = r_usage.ru_stime.tv_usec - sysStartTime.tv_usec;
    if (smsec < 0) {
      smsec += 1000000;
      ssec--;
    }

    setStopTime();
    os << notes << endl;
    os << "CPU Usage: user = " << usec
       << " seconds " << umsec << " ms, system = " << ssec 
       << " seconds " << smsec << "ms" << endl;
    os << "Time per iteration = "<<(usec+umsec/1e6)/iter<<endl;
    os << "ELAPSED Time: " << sec << " seconds "
       << microsec << " ms" << endl;
  }

  void getTotalElapsedSec(ostream & os){
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);

    os << "CPU Usage: user = " << r_usage.ru_utime.tv_sec
       << " seconds " << r_usage.ru_utime.tv_usec << " ms, system = " << r_usage.ru_stime.tv_sec
       << " seconds " << r_usage.ru_stime.tv_usec << "ms" << endl;
  }
  // Returns the seconds between calls to setStartTime and setEndTime
  long getElapsedSec() {return sec;}
  
  // Returns any extra microseconds after taking to account elapsed 
  // seconds
  long getElapsedMicrosec() {return microsec;}
   
};



#endif // _TIMERUTIL_H
