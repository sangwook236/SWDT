#!/usr/bin/env bash

# usage ---------------------------------------------------

# ---------------------------------------------------------

export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
export MOA_HOME=/usr/local/lib/moa

export PATH=/home/sangwook/work_center/sw_dev/java/ext/bin:$PATH

# ---------------------------------------------------------

export LIB_PATH1=/usr/local/lib

export CLASSPATH=.:$LIB_PATH1/moa.jar:$CLASSPATH

#export MAX_JAVA_MEMORY=800
#export HEAP_OPTION=-Xmx1000M

# ---------------------------------------------------------

java $HEAP_OPTION -javaagent:$LIB_PATH1\sizeofag-1.0.0.jar moa.gui.GUI
