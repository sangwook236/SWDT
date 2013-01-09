#include <iostream>

#ifdef _DEBUG
#define IL_DEBUG
#endif  // _DEBUG

#define ILUT_USE_OPENGL
#if defined(WiN32)
#include <IL/config.h>
#endif
#include <IL/il.h>
#include <IL/ilu.h>
#include <IL/ilut.h>


namespace {
namespace local {

void handleDevILErrors()
{
	ILenum error = ilGetError();

	if (IL_NO_ERROR != error)
	{
		do
		{
#if defined(UNICODE) || defined(_UNICODE)
			std::wcout << L'\t' << iluErrorString(error) << std::endl;
#else
			std::cout << '\t' << iluErrorString(error) << std::endl;
#endif
		} while (error = ilGetError());

		//exit(1);
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_devil {

void basic_operation();

}  // namespace my_devil

int devil_main(int argc, char *argv[])
{
	my_devil::basic_operation();

	local::handleDevILErrors();

	return 0;
}
