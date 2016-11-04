## Usage Guide

##### General
- Reference
	- http://www.mingw.org/wiki/MSVC_and_MinGW_DLLs
		* Important

	- http://www.mingw.org/wiki/Interoperability_of_Libraries_Created_by_Different_Compiler_Brands
	- http://stackoverflow.com/questions/25787344/using-dll-compiled-with-cygwin-inside-visual-studio-2010

- Note
	- Name mangling for stdcall functions is different in MSVC.
	- MSVC will prefix an underscore to __stdcall functions while MinGW will not.
		- [ref] http://www.mingw.org/wiki/MSVC_and_MinGW_DLLs

## Building

##### In MSVC
- Reference
	- http://www.mingw.org/wiki/MSVC_and_MinGW_DLLs
	- https://wiki.videolan.org/GenerateLibFromDll
	- http://cadcam.yonsei.ac.kr/bbs/zboard.php?id=resource&no=72

- GCC DLL -> import library
	1. Visual Studio Command Prompt를 실행.
		- Lib.exe를 사용하기 위해.
	2. export symbols from DLL
		- Dependency Walker
			- DLL 간의 dependency를 check해 주는 utility.
			- 또 다른 기능으로는 DLL에서 export하는 함수리스트를 보여 줌.
		- dumpbin
			- /exports option을 사용해 export된 함수리스트를 볼 수 있음.
				- dumpbin.exe /exports foo.dll > foo.dll.def
				- 생성된 export file을 그대로 DEF file로 사용할 수 없음.
					- DEF file 형식에 맞게 수정해야 함.
	3. modify DEF file.
		- e.g.) foo.dll.def

			`; LIBRARY foo.dll` <br />
			`EXPORTS` <br />
			`; 여기서 부터는 리스트된 함수를 나열.` <br />
			`; 예를 들어 func_a, func_b 라면` <br />
			`func_a` <br />
			`func_b`
	4. generate import library from DEF file.
		- lib.exe /def:foo.dll.def 
			- for 32-bit: `lib.exe /machine:i386 /def:foo.dll.def`
			- for 64-bit: `lib.exe /machine:X64 /def:foo.dll.def`
		- foo.dll.lib 생성.
		- 생성된 import library(lib)를 이용하여 link.

- DLL만 존재할 경우
	- LoadLibrary()를 사용하여 각 함수 prototype을 선언한 후 사용.

- DLL & header file을 가지고 있는 경우
	- DLL로 부터 import library를 만들어 일반적인 library와 동일하게 사용 가능.

- static library (*.a) in MinGW/Cygwin -> shared library (*.dll) in Windows
	- Cygwin에서 생성된 shared library file은 Windows에서 사용 가능하다는 점을 이용.
		- 하지만, Cygwin에서 생성된 static library (*.a)은 Windows에서 사용 불가. (???)
		- Cygwin에서 생성된 static library (*.a)로부터 shared library룰 생성한 후 Windows에서 사용.
			- [ref] http://www.terborg.net/research/kml/installation.html
				- kml_win_dll.sh 파일 참고.
				- Cygwin에 MinGW module들이 설치되어 있어야 정상적으로 작동.
	- MinGW에서도 동일하게 적용. (???)

- static library (*.a) in MinGW/Cygwin -> static library (*.lib) in Windows
	- [ref] http://stackoverflow.com/questions/2096519/from-mingw-static-library-a-to-visual-studio-static-library-lib
	- object file converter 사용.
		- [ref] http://www.agner.org/optimize/#objconv

##### In GCC(MinGW/Cygwin)
- Reference
	- http://www.mingw.org/wiki/MSVC_and_MinGW_DLLs

- MSVC DLL -> import library
	1. export symbols from DLL.
		- use nm

			`echo EXPORTS > foo.dll.def` <br />
			`nm foo.dll | grep ' T _' | sed 's/.* T _//' >> foo.dll.def`
		- use pexports in the mingw-utils package

			`pexports foo.dll | sed "s/^_//" > foo.dll.def`
		- Name mangling.
			- MSVC will prefix an underscore to __stdcall functions while MinGW will not.
			- [ref] http://www.mingw.org/wiki/MSVC_and_MinGW_DLLs
		- Note that this will only work if the DLL is not stripped.
			- Otherwise you will get an error message: "No symbols in foo.dll".
	2. generate import library from DLL & DEF file.

		`dlltool --def foo.dll.def --dllname foo.dll --output-lib foo.dll.a` <br />
		`dlltool -U -d foo.dll.def -l foo.dll.a`
		- Can generate import library when building DLL.

			`gcc -shared -o foo.dll foo.c -Wl,--output-def,foo.dll.def,--out-implib,foo.dll.a`
	3. (optional) strip import library (?)

		`strip foo.dll.a`
		- Removes unnecessary information from executable binary programs and object files,
			- thus potentially resulting in better performance and sometimes significantly less disk space usage.
