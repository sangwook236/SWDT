#!/usr/bin/env bash

# 32-bit or 64-bit

export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64

export PATH=$JAVA_HOME/bin:/home/sangwook/work_center/sw_dev/java/ext/bin:$PATH

# ---------------------------------------------------------

export LIB_PATH1=/usr/local/lib

#export CLASSPATH=.:$LIB_PATH1/javacpp.jar:$CLASSPATH

#export HEAP_OPTION=-Xms4096m -Xmx8192m

# build ---------------------------------------------------

javac -cp .:$LIB_PATH1/javacpp.jar $@.java
java $HEAP_OPTION -jar $LIB_PATH1/javacpp.jar $@

# run -----------------------------------------------------

#java $HEAP_OPTION -cp .:$LIB_PATH1/javacpp.jar $@
