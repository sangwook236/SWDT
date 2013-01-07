#!/usr/bin/env bash

# usage ---------------------------------------------------

# ---------------------------------------------------------

export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
export WEKA_HOME=/home/sangwook/work_center/sw_dev/java/rnd/src/machine_learning/weka/weka-3-7-7
export R_HOME=/usr/local/lib/R

export PATH=/home/sangwook/work_center/sw_dev/java/ext/bin:$PATH

# ---------------------------------------------------------

export LIB_PATH1=/usr/local/lib

export CLASSPATH=.:$LIB_PATH1/sqlite-jdbc-3.7.2.jar:$LIB_PATH1/mysql-connector-java-5.1.22-bin.jar:$LIB_PATH1/j3dcore.jar:$LIB_PATH1/j3dutils.jar:$CLASSPATH

#export MAX_JAVA_MEMORY=800
export HEAP_OPTION=-Xmx1000M

# ---------------------------------------------------------

java $HEAP_OPTION -jar $WEKA_HOME/weka.jar
