# distutils: language = c++

from libcpp cimport bool
import numpy as np
cimport numpy as np

assert sizeof(bool) == sizeof(np.uint8_t)

cdef extern from 'parallel_nms_cpu.cpp':
	pass

cdef extern from 'parallel_nms.h':
	cdef struct box:
		float x, y, w, h, s

	cdef void parallel_nms_best(box *b, int count, float threshold, bool *res)
	#cdef void parallel_nms_binary(box *b, int count, float threshold, bool *res)

def py_parallel_nms_cpu(np.ndarray[np.float32_t, ndim=2, mode='c'] boxes, np.float thresh):
	cdef int boxes_num = boxes.shape[0]
	cdef np.ndarray[np.uint8_t, ndim=1] res = np.ones(boxes_num, dtype=np.uint8)
	parallel_nms_best(<box *>&boxes[0,0], boxes_num, thresh, <bool *>&res[0])
	return list(res)
