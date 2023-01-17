TARGET	= libmatmult.so
LIBSRCS	= matmult_nat.c matmult_lib.c matmult_per.c matmult_blk.c
LIBOBJS	= matmult_nat.o matmult_lib.o matmult_per.o matmult_blk.o

OPT	= -g -O3 -funroll-all-loops
PIC	= -fPIC 

CC	= gcc
CFLAGS= $(OPT) $(PIC) $(XOPTS)

SOFLAGS = -shared 
XLIBS	= -L/usr/lib64/atlas -lsatlas

$(TARGET): $(LIBOBJS)
	$(CC) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS)
