#!/usr/bin/env bash

export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
#export RAPIDMINER_HOME=/usr/lib/Rapid-I/RapidMiner5
export RAPIDMINER_HOME=/home/sangwook/work_center/sw_dev/java/rnd/src/machine_learning/rapid_i/rapidminer-5.2.008/rapidminer

export PATH=/home/sangwook/work_center/sw_dev/java/ext/bin:$PATH

# ---------------------------------------------------------

export LIB_PATH1=/usr/local/lib

#export CLASSPATH=.:$LIB_PATH1/???.jar:$CLASSPATH

export MAX_JAVA_MEMORY=800
#export HEAP_OPTION=-Xms4096m -Xmx8192m

# ---------------------------------------------------------

java $HEAP_OPTION -Drapidminer.home=$RAPIDMINER_HOME -jar $RAPIDMINER_HOME/lib/rapidminer.jar
