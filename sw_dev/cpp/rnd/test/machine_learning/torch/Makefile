EXE_NAME=torch_test.exe
SRCS=torch_test.cpp
OBJS=$(SRCS:.cpp=.o)

CC=gcc
CXX=g++
LD=ld
AR=ar
RANLIB=ranlib
RM=rm -f

CFLAGS=
CXXFLAGS=-c -std=c++17 -Wall -I./libtorch/include -I/usr/local/include -I/usr/include
LDFLAGS=-L. -L./libtorch/lib -L/usr/local/lib -L/usr/lib -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10 -lcudart

#--------------------
all: $(SRCS) $(EXE_NAME)

$(EXE_NAME): $(OBJS) 
	$(CXX) -o $@ $^ $(LDFLAGS)

.cpp.o:
	$(CXX) -o $@ $< $(CXXFLAGS)

clean:
	$(RM) $(EXE_NAME) $(EXE_NAME).exe $(OBJS) *~ core
