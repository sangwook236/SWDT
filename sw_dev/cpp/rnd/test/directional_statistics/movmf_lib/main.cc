#include "movmf.h"

int main(int argc, char **argv)
{
  movmf* movmf_engine = new movmf(argc, argv);
  return movmf_engine->run();
}
