#!/usr/bin/env bash

# export JAVA_HOME=/usr/lib/jvm/java-6-openjdk-amd64

# export HADOOP_HOME=~/workspace/hadoop/hadoop-0.23.4
# export PATH=$PATH:$HADOOP_HOME/bin

# export HADOOP_CLASSPATH=.

javac -cp .:$HADOOP_HOME/share/hadoop/common/hadoop-common-0.23.4.jar:$HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-client-core-0.23.4.jar *.java

hadoop HadoopMain input/input_file.txt output
