#!/usr/bin/env bash

if [ $# -ne 1 ]; then

echo
echo "Usage:"
echo "    $0 [--set | --clean]"
echo

elif [ "$1" = "--set" ]; then

export HADOOP_HOME=/home/sangwook/spark-2.3.1-bin-hadoop2.7

export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
#export MAVEN_HOME=/home/sangwook/my_util/apache-maven-3.5.4
#export ANT_HOME=/home/sangwook/my_util/apache-ant-1.10.5
#export SBT_HOME=/home/sangwook/my_util/sbt-1.1.6/sbt
export SPARK_HOME=/home/sangwook/spark-2.3.1-bin-hadoop2.7

export SAVED_PATH_FOR_SPARK=$PATH
#export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$SBT_HOME/bin:$ANT_HOME/bin:$MAVEN_HOME/bin:$JAVA_HOME/bin:$PATH
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

elif [ "$1" = "--clean" ]; then

unset HADOOP_HOME
unset SPARK_HOME

export PATH=$SAVED_PATH_FOR_SPARK
unset SAVED_PATH_FOR_SPARK

else

echo
echo "Usage:"
echo "    $0 [--set | --clean]"
echo

fi

# ---------------------------------------------------------

