#include <algorithm>
#include "parallel_nms.h"

float IOUcalc(const box &b1, const box &b2)
{
	const float ai = float(b1.w + 1) * (b1.h + 1);
	const float aj = float(b2.w + 1) * (b2.h + 1);

	const float x_inter = std::max(b1.x, b1.x);
	const float y_inter = std::max(b1.y, b2.y);
	const float x2_inter = std::min((b1.x + b1.w), (b2.x + b2.w));
	const float y2_inter = std::min((b1.y + b1.h), (b2.y + b2.h));
	
	const float w = std::max(0.0f, x2_inter - x_inter);  
	const float h = std::max(0.0f, y2_inter - y_inter);  
	
	return ((w * h) / (ai + aj - w * h));
}

void parallel_nms_best(box *b, int count, float threshold, bool *res)
{
    //for (int i = 0; i < count; ++i)
    //	res[i] = true;

    for (int i = 0; i < count; ++i)
    	for (int j = 0; j < count; ++j)
    		if (b[i].s >= b[j].s && IOUcalc(b[i], b[j]) >= threshold)
 				res[j] = false; 
}
