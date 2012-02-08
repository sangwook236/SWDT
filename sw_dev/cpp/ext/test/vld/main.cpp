#include <vld/vld.h>
#include <iostream>


//----------------------------------------------------
// -. vld configuration file (vld.ini)은 executable file과 동일한 directory에 존재하여야 함.
//	  존재하지 않는 경우 기본 설정이 적용됨.
// -. 실행이 정상 종료되는 경우에 결과가 생성됨.
// -. 실행된 경로 상에 존재하는 memory leakage만을 detection.
// -. vld.h는 library or executable project의 하나의 file에서만 include 되면 됨. (?)
// -. 정상적인 실행을 위해서 vld_x86.dll & dbghelp.dll 필요.


#if defined(_UNICODE) || defined(UNICODE)
int wmain(int argc, wchar_t* argv[])
#else
int main(int argc, char* argv[])
#endif
{
	void basic(const bool leakage);
	void boost_thread(const bool leakage);

	try
	{
		const bool leakage = true;

		basic(leakage);
		boost_thread(leakage);
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::flush;
	std::cin.get();

    return 0;
}
