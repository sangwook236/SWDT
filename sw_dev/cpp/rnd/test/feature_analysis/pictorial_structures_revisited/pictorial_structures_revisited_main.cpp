#include <windows.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_pictorial_structures_revisited {

int pictorial_structures_revisited_partapp_main(int argc, char *argv[]);

}  // namespace my_pictorial_structures_revisited


//	[ref] ${PictorialStructureRevisited_HOME}/ReadMe.txt
//
//	-. to compute part posteriors for a single image
//			pictorial_structures_revisited.exe --expopt ./expopt/<EXP_FILENAME> --part_detect --find_obj --first <IMGIDX> --numimgs 1
//		examples of <EXP_FILENAME>
//			./expopt/exp_buffy_hog_detections.txt
//			./expopt/exp_ramanan_075.txt
//			./expopt/exp_tud_upright_people.txt
//
//	-. to precess the whole dataset
//			pictorial_structures_revisited.exe --expopt ./expopt/<EXP_FILENAME> --part_detect --find_obj
//
//	-. to evaluate the number of correctly detected parts
//			pictorial_structures_revisited.exe --expopt ./expopt/<EXP_FILENAME> --eval_segments --first <IMGIDX> --numimgs 1
//		this command will also produce visualization of the max-marginal part estimates in the "part_marginals/seg_eval_images" directory
//
//	-. to extract object hypothesis
//			pictorial_structures_revisited.exe --expopt ./expopt/<EXP_FILENAME> --save_res     
//		this will produce annotation files in the same format as training and test data. 
//
//	-. pretrained model (classifiers and joint parameters)
//		./log_dir/<EXP_NAME>/class
//
//	-. at runtime the following directories will be created:				      
//		./log_dir/<EXP_NAME>/test_scoregrid - location where part detections will be stored 
//		./log_dir/<EXP_NAME>/part_marginals - location where part marginals will be stored

int pictorial_structures_revisited_main(int argc, char *argv[])
{
#if 1
	const char currDir[MAX_PATH] = "./feature_analysis_data/pictorial_structures_revisited/code_test";
	const BOOL retval = SetCurrentDirectoryA(currDir);

	// testing
	const int my_argc = 5;
	const char *my_argv[5] = {
		argv[0],
		"--expopt", "./expopt/exp_code_test.txt",
		"--part_detect", "--find_obj"
	};
#elif 0
	const char currDir[MAX_PATH] = "feature_analysis_data/pictorial_structures_revisited/code_test";
	const BOOL retval = SetCurrentDirectoryA(currDir);

	// testing
	const int my_argc = 4;
	const char *my_argv[4] = {
		argv[0],
		"--expopt", "./expopt/exp_code_test.txt",
		"--eval_segments"
	};
#elif 0
	const char currDir[MAX_PATH] = "feature_analysis_data/pictorial_structures_revisited/partapp-experiments-r2";
	const BOOL retval = SetCurrentDirectoryA(currDir);

	// experiment
	const int my_argc = 9;
	const char *my_argv[9] = {
		argv[0],
		"--expopt", "./expopt/exp_buffy_hog_detections.txt",
		//"--expopt", "./expopt/exp_ramanan_075.txt",
		//"--expopt", "./expopt/exp_tud_upright_people.txt",
		"--part_detect", "--find_obj",
		"--first", "<IMGIDX>",
		"--numimgs", "1"
	};
#elif 0
	const char currDir[MAX_PATH] = "feature_analysis_data/pictorial_structures_revisited/partapp-experiments-r2";
	const BOOL retval = SetCurrentDirectoryA(currDir);

	// experiment
	const int my_argc = 5;
	const char *my_argv[5] = {
		argv[0],
		"--expopt", "./expopt/exp_buffy_hog_detections.txt",
		//"--expopt", "./expopt/exp_ramanan_075.txt",
		//"--expopt", "./expopt/exp_tud_upright_people.txt",
		"--part_detect", "--find_obj"
	};
#elif 0
	const char currDir[MAX_PATH] = "feature_analysis_data/pictorial_structures_revisited/partapp-experiments-r2";
	const BOOL retval = SetCurrentDirectoryA(currDir);

	// experiment
	const int my_argc = 8;
	const char *my_argv[8] = {
		argv[0],
		"--expopt", "./expopt/exp_buffy_hog_detections.txt",
		//"--expopt", "./expopt/exp_ramanan_075.txt",
		//"--expopt", "./expopt/exp_tud_upright_people.txt",
		"--eval_segments",
		"--first", "<IMGIDX>",
		"--numimgs", "1"
	};
#elif 0
	const char currDir[MAX_PATH] = "feature_analysis_data/pictorial_structures_revisited/partapp-experiments-r2";
	const BOOL retval = SetCurrentDirectoryA(currDir);

	// experiment
	const int my_argc = 4;
	const char *my_argv[4] = {
		argv[0],
		"--expopt", "./expopt/exp_buffy_hog_detections.txt",
		//"--expopt", "./expopt/exp_ramanan_075.txt",
		//"--expopt", "./expopt/exp_tud_upright_people.txt",
		"--eval_segments"
	};
#endif

	std::cout << "-----------------------------------------" << std::endl;
	for (int i = 0; i < my_argc; ++i)
		std::cout << "argv[" << i << "] : " << my_argv[i] << std::endl;

	const char *home = getenv("HOME");
	if (home)
		std::cout << "environment variable, HOME = " << home << std::endl;
	else
	{
		std::cout << "environment variable, HOME, is not found" << std::endl;
		return -1;
	}
	std::cout << "-----------------------------------------" << std::endl;

	my_pictorial_structures_revisited::pictorial_structures_revisited_partapp_main(my_argc, (char **)my_argv);

	return 0;
}
