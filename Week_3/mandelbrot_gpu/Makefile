TARGET= mandelbrot
OBJS	= main.o \
        writepng.o \
        mandel_omp.o \

OPT	= -g -fast -Msafeptr -Minfo -acc -mp=gpu -gpu=pinned -gpu=cc80 -gpu=lineinfo -cuda -mp=noautopar
ISA	= 
PARA	= 

PNGWRITERPATH = pngwriter
ARCH	      = $(shell uname -p)
PNGWRTLPATH   = $(PNGWRITERPATH)/lib/$(ARCH)
PNGWRTIPATH   = $(PNGWRITERPATH)/include
PNGWRITERLIB  = $(PNGWRTLPATH)/libpngwriter.a

CC	= nvc

CXX	= nvc++
CFLAGS= -I $(PNGWRTIPATH) $(OPT) $(ISA) $(PARA) $(XOPT)
CXXFLAGS= -I $(PNGWRTIPATH) $(OPT) $(ISA) $(PARA) $(XOPT)

F90C  	= gfortran
LIBS	= -L $(PNGWRTLPATH) -lpngwriter -lpng 

all: $(PNGWRITERLIB) $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

$(PNGWRITERLIB):
	@cd pngwriter/src && $(MAKE) -f $(MAKEFILE_LIST)

clean:
	@/bin/rm -f *.o $(TARGET)

realclean: clean
	@cd pngwriter/src && $(MAKE) -f $(MAKEFILE_LIST) clean
	@rm -f $(PNGWRITERLIB)
	@rm -f $(TARGET)
	@rm -f mandelbrot.png

# dependencies
#
main.o  : main.cpp mandel_omp.h
mandel_omp.o: mandel_omp.cpp
writepng.o: writepng.h writepng.cc
