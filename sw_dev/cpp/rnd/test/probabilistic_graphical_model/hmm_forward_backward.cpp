//#include "stdafx.h"
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

// [ref] umdhmm (http://www.kanungo.com/software/software.html#umdhmm)
//	forward.c & testfor.c
void hmm_forward_umdhmm()
{
	throw std::runtime_error("not yet implemented");
}

// [ref] umdhmm (http://www.kanungo.com/software/software.html#umdhmm)
//	backward.c
void hmm_backward_umdhmm()
{
	throw std::runtime_error("not yet implemented");
}

}  // namespace local
}  // unnamed namespace

void hmm_forward_backward()
{
    local::hmm_forward_umdhmm();
    local::hmm_backward_umdhmm();
}

