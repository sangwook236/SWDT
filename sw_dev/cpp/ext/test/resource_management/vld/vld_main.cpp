#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


//----------------------------------------------------
// -. vld configuration file (vld.ini)은 executable file과 동일한 directory에 존재하여야 함.
//	  존재하지 않는 경우 기본 설정이 적용됨.
// -. 실행이 정상 종료되는 경우에 결과가 생성됨.
// -. 실행된 경로 상에 존재하는 memory leakage만을 detection.
// -. vld.h는 library or executable project의 하나의 file에서만 include 되면 됨. (?)
// -. 정상적인 실행을 위해서 vld_x86.dll & dbghelp.dll 필요.


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vld {

void basic(const bool leakage);
void boost_thread(const bool leakage);

}  // namespace my_vld

int vld_main(int argc, char *argv[])
{
	const bool leakage = true;

	my_vld::basic(leakage);
	my_vld::boost_thread(leakage);

    return 0;
}
