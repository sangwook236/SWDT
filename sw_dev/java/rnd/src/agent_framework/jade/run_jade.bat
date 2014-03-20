@echo off

echo Running %*%

if "%JADE_CLASSPATH%" == "" goto DEFAULT

rem -----------------------------------------------------------
echo Using JADE_CLASSPATH variable to access bundles at %JADE_CLASSPATH%
java -classpath %JADE_CLASSPATH% jade.Boot %*%
goto EXIT

rem -----------------------------------------------------------
:DEFAULT
echo Using JADE_CLASSPATH bundles installed as standard extensions
java jade.Boot %*%

rem -----------------------------------------------------------
:EXIT

echo on
