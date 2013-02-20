#include "QPBO.h"

#ifdef _MSC_VER
#pragma warning(disable: 4661)
#endif

// Instantiations
namespace kolmogorov{
namespace qpbo{

template class QPBO<int>;
template class QPBO<float>;
template class QPBO<double>;

template <> 
	inline void QPBO<int>::get_type_information(char*& type_name, char*& type_format)
{
	type_name = (char*)("int");
	type_format = (char*)("d");
}

template <> 
	inline void QPBO<float>::get_type_information(char*& type_name, char*& type_format)
{
	type_name = (char*)("float");
	type_format = (char*)("f");
}

template <> 
	inline void QPBO<double>::get_type_information(char*& type_name, char*& type_format)
{
	type_name = (char*)("double");
	type_format = (char*)("Lf");
}

}}
