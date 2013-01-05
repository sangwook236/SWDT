#!/usr/bin/env bash

export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64

export PATH=/home/sangwook/work_center/sw_dev/java/ext/bin:$PATH

# ---------------------------------------------------------

#export LIB_PATH1=/usr/local/lib

#export CLASSPATH=.:$LIB_PATH1/runnable_jar.jar:$CLASSPATH

#export MAX_JAVA_MEMORY=800
#export HEAP_OPTION=-Xms4096m -Xmx8192m

# ---------------------------------------------------------

#java $HEAP_OPTION -jar $LIB_PATH1/runnable_jar.jar java_jar.Hello
java $HEAP_OPTION -jar $LIB_PATH1/runnable_jar.jar java_jar.Hi

# error ---------------------------------------------------
#java $HEAP_OPTION -jar $LIB_PATH/nonrunnable_jar.jar
