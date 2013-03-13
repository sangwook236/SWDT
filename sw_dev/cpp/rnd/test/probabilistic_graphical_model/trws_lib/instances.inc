
#include "MRFEnergy.h"

#if 0
template MRFEnergy<TypeBinary>;
template MRFEnergy<TypeBinaryFast>;
template MRFEnergy<TypePotts>;
template MRFEnergy<TypeGeneral>;
template MRFEnergy<TypeTruncatedLinear>;
template MRFEnergy<TypeTruncatedQuadratic>;
template MRFEnergy<TypeTruncatedLinear2D>;
template MRFEnergy<TypeTruncatedQuadratic2D>;
#else
template class MRFEnergy<TypeBinary>;
template class MRFEnergy<TypeBinaryFast>;
template class MRFEnergy<TypePotts>;
template class MRFEnergy<TypeGeneral>;
template class MRFEnergy<TypeTruncatedLinear>;
template class MRFEnergy<TypeTruncatedQuadratic>;
template class MRFEnergy<TypeTruncatedLinear2D>;
template class MRFEnergy<TypeTruncatedQuadratic2D>;
#endif
