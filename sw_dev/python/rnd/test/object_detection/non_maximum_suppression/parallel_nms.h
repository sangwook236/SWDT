#if !defined(__PARALLEL_NMS_H_)
#define __PARALLEL_NMS_H_ 1

struct box
{
	float x, y, w, h, s;
};

// REF [site] >> https://github.com/jeetkanjani7/Parallel_NMS/blob/master/cpu/nms.cpp
void parallel_nms_best(box *b, int count, float threshold, bool *res);
//void parallel_nms_binary(box *b, int count, float threshold, bool *res);
// REF [site] >> https://github.com/jeetkanjani7/Parallel_NMS/blob/master/GPU/nms.cu
void parallel_nms_gpu(box *b, int count, float threshold, bool *res);

#endif  // __PARALLEL_NMS_H_
