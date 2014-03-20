@echo off

echo Compiling %*%

if "%JADE_CLASSPATH%" == "" goto DEFAULT

rem -----------------------------------------------------------
echo Using JADE_CLASSPATH variable to access bundles at %JADE_CLASSPATH%
javac -classpath %JADE_CLASSPATH% %*%
goto EXIT

rem -----------------------------------------------------------
:DEFAULT
echo Using JADE_CLASSPATH bundles installed as standard extensions
javac %*%

rem -----------------------------------------------------------
:EXIT

echo on
