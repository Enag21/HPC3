TARGET= matvec
OBJS  = main.o \
        mxv.o \
        alloc.o \

CC	= nvc
CXX	= nvc++

OPT	= -g -fast -Msafeptr -Minfo -acc -mp=gpu -gpu=pinned -gpu=cc80 -gpu=lineinfo -cuda -mp=noautopar
ISA	= 
PARA	= 
INC   = -I$(CUDA_ROOT)/include -I/appl/nvhpc/2022_2211/Linux_x86_64/22.11/examples/OpenMP/SDK/include
LIBS	= -lcuda -L$(CUDA_ROOT)/lib64 -lcudart #-lcublas

CFLAGS = $(OPT) $(INC) $(ISA) $(PARA) $(XOPT)
CXXFLAGS = $(OPT) $(INC) $(ISA) $(PARA) $(XOPT)

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)
clean:
	@/bin/rm -f *.o $(TARGET)
