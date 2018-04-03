#!/usr/bin/env bash

# usage ---------------------------------------------------

# ---------------------------------------------------------

export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
export WEKA_HOME=/home/sangwook/my_util/weka-3-9-2
export R_HOME=/usr/local/lib/R

export PATH=/home/sangwook/work/SWDT_github/sw_dev/java/ext/bin:$PATH

# ---------------------------------------------------------

export LIB_PATH=/usr/local/lib

export CLASSPATH=.:$LIB_PATH/sqlite-jdbc-3.7.2.jar:$LIB_PATH/mysql-connector-java-5.1.22-bin.jar:$LIB_PATH/j3dcore.jar:$LIB_PATH/j3dutils.jar:$CLASSPATH

#export MAX_JAVA_MEMORY=800
export HEAP_OPTION=-Xmx1000M

# ---------------------------------------------------------

java $HEAP_OPTION -jar $WEKA_HOME/weka.jar
