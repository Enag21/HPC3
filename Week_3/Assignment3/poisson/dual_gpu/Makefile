TARGET= poisson
FF=$(shell pwd)
OBJS	= main.o \
        jacobi.o \
        init.o \
        print.o \
        alloc3d.o \
        #input.o \

CC	= nvc
CXX	= nvc++

OPT	= -g -fast -Msafeptr -Minfo -acc -mp=gpu -gpu=pinned -gpu=cc80 -gpu=lineinfo -cuda -mp=noautopar
ISA	= 
PARA	= 
INC   = -I$(CUDA_ROOT)/include -I/appl/nvhpc/2022_2211/Linux_x86_64/22.11/examples/OpenMP/SDK/include -I$(FF)$
LIBS	= -lcuda

CFLAGS = $(OPT) $(INC) $(ISA) $(PARA) $(XOPT)
CXXFLAGS = $(OPT) $(INC) $(ISA) $(PARA) $(XOPT)

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)
clean:
	@/bin/rm -f *.o $(TARGET)
