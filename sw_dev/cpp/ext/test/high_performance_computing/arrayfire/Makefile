SRCS=arrayfire_morphology.cpp
OBJS=$(SRCS:.cpp=.o)
EXE_NAME=test

ARRYFIRE_HOME=/home/sangwook/lib_repo/cpp/arrayfire

CC=
CXX=g++
LD=ld
AR=ar
RANLIB=ranlib
RM=rm -f

#-- Linux, Cygwin, & MinGW(MSYS).
CFLAGS=
CXXFLAGS=-c -Wall -I$(ARRYFIRE_HOME)/include
LDFLAGS=-L. -L$(ARRYFIRE_HOME)/lib64 -L/usr/local/cuda/lib64 -lafopencl -lcusolver -lcudart -lcufft -lcublas -lmkl_core
#LDFLAGS=-L. -L$(ARRYFIRE_HOME)/lib64 -L/usr/local/cuda/lib64 -lafcuda -lcusolver -lcudart -lcufft -lcublas -lmkl_core
#LDFLAGS=-L. -L$(ARRYFIRE_HOME)/lib64 -L/usr/local/cuda/lib64 -lafcpu -lcusolver -lcudart -lcufft -lcublas -lmkl_core

all: $(SRCS) $(EXE_NAME)

$(EXE_NAME): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

.cpp.o:
	$(CXX) -o $@ $< $(CXXFLAGS)

clean:
	$(RM) $(EXE_NAME) $(EXE_NAME).exe *.o *~ core
