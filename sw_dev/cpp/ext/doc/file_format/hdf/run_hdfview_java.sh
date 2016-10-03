#!/usr/bin/env bash

# usage ---------------------------------------------------

# ---------------------------------------------------------

export JAVAHOME=/usr/lib/jvm/java-7-openjdk-amd64

export HDFVIEW_HOME=/usr/local/bin/hdf-java-2.9-bin\hdf-java

export PATH=$HDFVIEW_HOME/bin:$JAVAHOME/bin:$PATH

# ---------------------------------------------------------

export CLASSPATH=$HDFVIEW_INSTALL/*:$HDFVIEW_INSTALL/ext/*
export LIB_PATH=$HDFVIEW_INSTALL:$HDFVIEW_INSTALL/ext

export HEAP_OPTION=-Xmx1024m

# ---------------------------------------------------------

java $HEAP_OPTION -Djava.library.path=$LIB_PATH -Dhdfview.root=$HDFVIEW_INSTALL ncsa.hdf.view.HDFView -root $HDFVIEW_INSTALL
