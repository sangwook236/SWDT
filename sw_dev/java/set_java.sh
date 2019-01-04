#!/usr/bin/env bash

if [ $# -ne 1 ]; then

echo
echo "Usage:"
echo "    $0 [--set | --clean]"
echo

elif [ "$1" = "--set" ]; then

export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64

export SAVED_PATH_FOR_JAVA=$PATH
export PATH=$JAVA_HOME/bin:$PATH

elif [ "$1" = "--clean" ]; then

unset JAVA_HOME

export PATH=$SAVED_PATH_FOR_JAVA
unset SAVED_PATH_FOR_JAVA

else

echo
echo "Usage:"
echo "    $0 [--set | --clean]"
echo

fi
