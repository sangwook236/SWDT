[-] Setting (C/C++)
	-. Toolchain
		"Settings" main menu item -> "Compiler..." menu item -> "Global compiler settings" icon -> "Toolchain executables" tab
		"Settings" main menu item -> "Compiler..." menu item -> "Global compiler settings" icon -> "Compiler settings" and "Linker settings" tabs

	-. Default include & library directories
		"Settings" main menu item -> "Compiler..." menu item -> "Global compiler settings" icon -> "Toolchain executables" tab (?)
			==> This is not a fundamental solution, but changes priorities of search directories.

	-. Compiler
		Additional include directories
			"Project" main menu item -> "Build options..." menu item -> "Search directories" tab -> "Compiler" tab
		Compiler options
			"Project" main menu item -> "Build options..." menu item -> "Compiler settings" tab -> "Compiler Flags" & "Other options" tabs
		Defines
			"Project" main menu item -> "Build options..." menu item -> "Compiler settings" tab -> "#defines" tab

	-. Linker
		Additional library directories
			"Project" main menu item -> "Build options..." menu item -> "Search directories" tab -> "Linker" tab
		Link libraries
			"Project" main menu item -> "Build options..." menu item -> "Linker settings" tab -> "Link libraries:" listbox
		Link linker options
			"Project" main menu item -> "Build options..." menu item -> "Linker settings" tab -> "Other linker options:" listbox

	-. Output directory
		"Project" main menu item -> "Properties..." menu item -> "Build targets" tab -> "Output filename:" item

	-. Working directory
		"Project" main menu item -> "Properties..." menu item -> "Build targets" tab -> "Execution working dir:" item

		Change working directory using cd command in script (in Mac).
			"Settings" main menu item -> "Environment..." menu item -> "General settings" listview item -> "Terminal to launch console programs:" item
				osascript -e 'tell app "Terminal"' -e 'activate' -e 'do script "exe=\'$SCRIPT\'; cd \\"${exe%/*/*/*}\\"; pwd; \\"${exe%}\\""' -e 'end tell'
					Where is $SCRIPT defined?
					'/*' means deletion of a word.
				[ref] AppleScript
			[ref] http://forums.codeblocks.org/index.php?topic=10328.0

			For building & running.
				osascript -e 'tell app "Terminal"' -e 'activate' -e 'do script "cd /path/to/working_directory; $SCRIPT"' -e 'end tell'
			For debugging.
				osascript -e 'tell app "Terminal"' -e 'activate' -e 'do script "cd /path/to/working_directory; ./executable_name"' -e 'end tell'

			==> I don't know exactly yet how to use AppleScript. This is just a starting point. 

	-. Project dependency
		"Project" main menu item -> "Properties..." menu item -> "Project settings" tab -> "Project's dependencies..." button

	-. Debugger
		"Settings" main menu item -> "Debugger..." menu item -> "Common" & "GDB/CDB debugger" tree items
		"Project" main menu item -> "Properties..." menu item -> "Debugger" tab

	-. Profiler

[-] Trouble Shooting
	-. 공용 library가 아닌 library를 link하기
		아래와 같이 library를 지정하여야 함.
			library 이름만 지정해서는 정상적으로 build되지 않음.
			../../bin/Debug/swl_base.so
			../../bin/Release/swl_base.so
			=> 이 경우 Linker search directories에 아래의 directories를 추가할 필요가 없음.
				../../bin/Debug/
				../../bin/Release/
	
	-. static library & shared object의 경우
		"Build options..."에서 Link libraries & Linker search directories를 설정할 필요가 없음.
			=> static library & shared object 모두 실제 linking 과정을 수행하지 않는 듯함.
				Linker와 관련된 option은 설정하기 않아도 될 듯함.
		Compiler search directories는 설정해야 함.
	
	-. options for build targets
		Compiler settings & Linker settings, Compiler search directories & Linker search directories
			전체 build target에 대해 설정된 option들은 build target (Release/Debug)으로 복사하지 않아도 됨.
		단. Link libraries 중 일부의 경우 build target (Release/Debug)으로 복사해야 정상적으로 link되는 경우가 있음.
	
	-. policy for target options
		Compiler settings & Linker settings
			적절한 compiler options & linker options을 적용하기 위해
			"Compiler settings"과 "Linker settings"에서 target option을 추가하는 policy를
			"Prepend target options to project options"을 지정해야 함.
			기본값은 "Append target options to project options"임.
			=> 특히, "Link libraries"의 경우 build 과정에 영향을 미칠 수 있음으로 중요.
	
		Compiler search directories & Linker search directories
			적절한 include directories & library directories를 사용하기 위해
			"Search directories"에서 target option을 추가하는 policy를
			"Prepend target options to project options"을 지정해야 함.
			기본값은 "Append target options to project options"임.
