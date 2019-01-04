@echo off
setlocal

java.exe -jar d:\util_center\software_dev_tool\smc\smc_6_1_0\bin\smc.jar -csharp -sync %1%
java.exe -jar d:\util_center\software_dev_tool\smc\smc_6_1_0\bin\smc.jar -table %1%
java.exe -jar d:\util_center\software_dev_tool\smc\smc_6_1_0\bin\smc.jar -graph -glevel 2 %1%

endlocal
echo on
