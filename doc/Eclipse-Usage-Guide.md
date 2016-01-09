## Setting (Java)
- Toolchain

- Default library directories

- Compiler
	- Compiler options

- Linker
	- Additional library directories
	- Link libraries
	- Link linker options

- Output directory

- Working directory

- Project dependency

- Debugger

## Setting (C/C++)
- Toolchain
	- "Project" main menu item -> "Properties" menu item -> "C/C++ Build" tree item -> "Tool Chain Editor" tree item (?)
		- Cannot assign a tool chain which I want to use, but just choose one of existing ones.
		- [ref] http://www.codebuilder.me/2014/03/setting-up-eclipse-on-a-mac-for-gcc-4-8-toolchain/ (?)
	- "Project" main menu item -> "Properties" menu item -> "C/C++ Build" tree item -> "Settings" tree item -> "Tool Settings" tab -> "xxx Compiler" tree item -> "Command:" item (?)
		- e.g.) g++ : `/my_usr/local/bin/g++`
	- "Project" main menu item -> "Properties" menu item -> "C/C++ Build" tree item -> "Settings" tree item -> "Tool Settings" tab -> "xxx Linker" tree item -> "Command:" item (?)
		- e.g.) g++ : `/mu_usr/local/bin/g++`
		- These are not a fundamental solution, but change just executable compiler & linker.

	- How to add custom tool chain to eclipse CDT.
		- http://stackoverflow.com/questions/3489607/how-to-add-custom-tool-chain-to-eclipse-cdt
		- http://sourceforge.net/projects/gnuarmeclipse/
		- http://gnuarmeclipse.livius.net/blog/toolchain-path/
		- These are not a solution but a guideline.

- Default include & library directories
	- Use 'Additional include directories' & 'Additional library directories'.
		- This is not a fundamental solution, but changes priorities of search directories.

- Compiler
	- Additional include directories
		- "Project" main menu item -> "Properties" menu item -> "C/C++ General" tree item -> "Code Analysis" tree item -> "Paths and Symbols" tree item -> "Includes" tab
			- set directory paths as usual.
				- e.g.) `../../inc`
		- "Project" main menu item -> "Properties" menu item -> "C/C++ Build" tree item -> "Settings" tree item -> "Tool Settings" tab -> "xxx Compiler" tree item -> "Includes" tree item -> "Include paths (-I)" listview
			- set directory paths with a deeper step than we think.
				- e.g.) `../../../inc`
	- Compiler options
		- "Project" main menu item -> "Properties" menu item -> "C/C++ Build" tree item -> "Settings" tree item -> "Tool Settings" tab -> "xxx Compiler" tree item -> "Optimization", "Debugging", "Warnings", and "Miscellaneous" tree items
	- Defines
		- "Project" main menu item -> "Properties" menu item -> "C/C++ General" tree item -> "Code Analysis" tree item -> "Paths and Symbols" tree item -> "Symbols" tabs
		- "Project" main menu item -> "Properties" menu item -> "C/C++ Build" tree item -> "Settings" tree item -> "Tool Settings" tab -> "xxx Compiler" tree item -> "Preprocessor" tree item -> "Defined symbols (-D)" listview

- Linker
	- Additional library directories
		- "Project" main menu item -> "Properties" menu item -> "C/C++ General" tree item -> "Code Analysis" tree item -> "Paths and Symbols" tree item -> "Library Paths" tab
			- set directory paths as usual.
				- e.g.) `../../lib`
		- "Project" main menu item -> "Properties" menu item -> "C/C++ Build" tree item -> "Settings" tree item -> "Tool Settings" tab -> "xxx Linker" tree item -> "Libraries" tree item -> "Library search path (-L)" listview
			- set directory paths with a deeper step than we think.
				- e.g.) `../../../lib`
	- Link libraries
		- "Project" main menu item -> "Properties" menu item -> "C/C++ General" tree item -> "Code Analysis" tree item -> "Paths and Symbols" tree item -> "Libraries" tab
		- "Project" main menu item -> "Properties" menu item -> "C/C++ Build" tree item -> "Settings" tree item -> "Tool Settings" tab -> "xxx Linker" tree item -> "Libraries" tree item -> "Libraries (-l)" listview
	- Link linker options
		- "Project" main menu item -> "Properties" menu item -> "C/C++ Build" tree item -> "Settings" tree item -> "Tool Settings" tab -> "xxx Linker" tree item -> "General", "Miscellaneous", and "Shared Library Settings" tree items

- Output directory
	- "Project" main menu item -> "Properties" menu item -> "C/C++ Build" tree item -> "Settings" tree item -> "Build Artifact" tab -> "Output prefix:" item (?)
		- in case of library : `../../../lib/lib`

- Working directory
	- "Run" main menu item -> "Run/Debug/Profile Configurations..." menu item -> "C/C++ Application" tree item -> 'project launch configuration' tree item -> "(x)= Arguments" tab -> "Working directory:" item
		- e.g.) `${workspace_loc:project_launch_configuration}/../../bin`

- Project dependency
	- "Project" main menu item -> "Properties" menu item -> "Project References" tree item
	- "Project" main menu item -> "Properties" menu item -> "C/C++ General" tree item -> "Code Analysis" tree item -> "Paths and Symbols" tree item -> "References" tab

- Debugger
	- "Run" main menu item -> "Debug Configurations..." menu item -> "C/C++ Application" tree item -> 'project launch configuration' tree item -> "Debugger" tab
	- Main menu item -> "Preferences..." menu item -> "C/C++" tree item -> "Debug" tree item -> "GDB" tree item

- Profiler

## Trouble Shooting
