#!/usr/bin/env bash

# Usage ---------------------------------------------------

# ---------------------------------------------------------

export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64

export PATH=/home/sangwook/work_center/sw_dev/java/ext/bin:$PATH

# ---------------------------------------------------------

export LOCAL_LIB_PATH=/usr/local/lib

#export CLASSPATH=.:$LOCAL_LIB_PATH/<jar-file>:$CLASSPATH

#export MAX_JAVA_MEMORY=800
#export HEAP_OPTION=-Xms4096m -Xmx8192m

# ---------------------------------------------------------

java $HEAP_OPTION -jar $LOCAL_LIB_PATH/<runnable-jar-file> <class-name>

# Error ---------------------------------------------------
#java $HEAP_OPTION -jar $LIB_PATH/<non-runnable-jar-file>
