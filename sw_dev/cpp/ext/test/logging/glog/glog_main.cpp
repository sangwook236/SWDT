#include <glog/logging.h>
#include <string>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_glog {

}  // namespace my_glog

// REF [site] >> https://google-glog.googlecode.com/svn/trunk/doc/glog.html
/*
	If you want to turn the flag --logtostderr on, you can start your application with the following command line:
        ./your_application --logtostderr=1
    If the Google gflags library isn't installed, you set flags via environment variables, prefixing the flag name with "GLOG_", e.g.
        GLOG_logtostderr=1 ./your_application

    The following flags are most commonly used:
    logtostderr (bool, default=false)
        Log messages to stderr instead of logfiles.
        Note: you can set binary flags to true by specifying 1, true, or yes (case insensitive). Also, you can set binary flags to false by specifying 0, false, or no (again, case insensitive).
    stderrthreshold (int, default=2, which is ERROR)
        Copy log messages at or above this level to stderr in addition to logfiles. The numbers of severity levels INFO, WARNING, ERROR, and FATAL are 0, 1, 2, and 3, respectively.
    minloglevel (int, default=0, which is INFO)
        Log messages at or above this level. Again, the numbers of severity levels INFO, WARNING, ERROR, and FATAL are 0, 1, 2, and 3, respectively.
    log_dir (string, default="")
        If specified, logfiles are written into this directory instead of the default logging directory.
    v (int, default=0)
        Show all VLOG(m) messages for m less or equal the value of this flag. Overridable by --vmodule. See the section about verbose logging for more detail.
    vmodule (string, default="")
        Per-module verbose level. The argument has to contain a comma-separated list of <module name>=<log level>. <module name> is a glob pattern (e.g., gfs* for all modules whose name starts with "gfs"), matched against the filename base (that is, name ignoring .cc/.h./-inl.h). <log level> overrides any value given by --v. See also the section about verbose logging.
*/
int glog_main(int argc, char *argv[])
{
    // Initialize Google's logging library.
	google::InitGoogleLogging(argv[0]);

    const int num_cookies = 37;
	LOG(INFO) << "Found " << num_cookies << " cookies";

    // Setting flags.
    {
        LOG(INFO) << "file";
        // Most flags work immediately after updating values.
        FLAGS_logtostderr = 1;
        LOG(INFO) << "stderr";
        FLAGS_logtostderr = 0;

        // This won't change the log destination.
        // If you want to set this value, you should do this before google::InitGoogleLogging.
        FLAGS_log_dir = "/some/log/directory";
        LOG(INFO) << "the same file";
    }

    // Conditional / occasional logging.
    {
        // The "Got lots of cookies" message is logged only when the variable num_cookies exceeds 10.
        LOG_IF(INFO, num_cookies > 10) << "Got lots of cookies";

        // The above line outputs a log messages on the 1st, 11th, 21st, ... times it is executed.
        // Note that the special google::COUNTER value is used to identify which repetition is happening.
        LOG_EVERY_N(INFO, 10) << "Got the " << google::COUNTER << "th cookie";

        // Instead of outputting a message every nth time, you can also limit the output to the first n occurrences.
        const int size = 1025;
        LOG_IF_EVERY_N(INFO, (size > 1024), 10) << "Got the " << google::COUNTER << "th big cookie";

        // Outputs log messages for the first 20 times it is executed.
        LOG_FIRST_N(INFO, 20) << "Got the " << google::COUNTER << "th cookie";
    }

    // Debug mode support.
    {
        DLOG(INFO) << "Found cookies";
        DLOG_IF(INFO, num_cookies > 10) << "Got lots of cookies";
        DLOG_EVERY_N(INFO, 10) << "Got the " << google::COUNTER << "th cookie";
    }

    // CHECK macros.
    {
        // CHECK aborts the application if a condition is not true.
        // Unlike assert, it is *not* controlled by NDEBUG, so the check will be executed regardless of compilation mode.
        // CHECK_EQ, CHECK_NE, CHECK_LE, CHECK_LT, CHECK_GE, CHECK_GT.
        // CHECK_STREQ, CHECK_STRNE, CHECK_STRCASEEQ, and CHECK_STRCASENE.
        // CHECK_DOUBLE_EQ.
        const int retval = 4;
        CHECK(retval == 4) << "Write failed!";

        CHECK_NE(1, 2) << ": The world must be ending!";
        CHECK_EQ(std::string("abc")[1], 'b');

        int *some_ptr = new int (4321);
        //int *some_ptr = NULL;
        CHECK_NOTNULL(some_ptr);
        std::cout << *some_ptr << std::endl;
        delete some_ptr;
    }

    // Verbose logging.
    {
        // if --v==1, VLOG(1) will log, but VLOG(2) will not log.
        VLOG(1) << "I'm printed when you run the program with --v=1 or higher";
        VLOG(2) << "I'm printed when you run the program with --v=2 or higher";

        // VLOG_IS_ON(n) returns true when the --v is equal or greater than n.
        if (VLOG_IS_ON(2))
        {
            // do some logging preparation and logging that can't be accomplished with just VLOG(2) << ...;
        }

        const int size = 1025;
        VLOG_IF(1, (size > 1024)) << "I'm printed when size is more than 1024 and when you run the program with --v=1 or more";
        VLOG_EVERY_N(1, 10) << "I'm printed every 10th occurrence, and when you run the program with --v=1 or more. Present occurence is " << google::COUNTER;
        VLOG_IF_EVERY_N(1, (size > 1024), 10) << "I'm printed on every 10th occurence of case when size is more than 1024, when you run the program with --v=1 or more. Present occurence is " << google::COUNTER;
    }

	return 0;
}
