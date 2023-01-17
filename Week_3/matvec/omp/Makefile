CC	= gcc
DEBUG = -g -Wall
OPT	=-fopenmp
POPT =  -O3 -ffast-math -ftree-parallelize-loops=4
AUTOPAR = -g -fast -xautopar -xreduction -xloopinfo
CFLAGS= $(OPT) $(DEBUG) 

SOURCES = main.c
OBJS = $(SOURCES:.c=.o)

all: parallel 

parallel: $(OBJS)
	$(CC) -o $@ $(CFLAGS) $(POPT) $(OBJS) 


clean:
	@/bin/rm -f $(OBJS) core

realclean: clean
	@/bin/rm -f serial parallel omp
