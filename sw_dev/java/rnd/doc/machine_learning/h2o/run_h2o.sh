#!/usr/bin/env bash

# usage ---------------------------------------------------

# ---------------------------------------------------------

export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
export WEKA_HOME=/home/sangwook/work/SWDT_github/sw_dev/java/rnd/src/machine_learning/weka/weka-3-7-7

export PATH=/home/sangwook/work/SWDT_github/sw_dev/java/ext/bin:$PATH

# ---------------------------------------------------------

export CLASSPATH=.:$CLASSPATH

# ---------------------------------------------------------

java -jar $H2O_HOME/h2o.jar
