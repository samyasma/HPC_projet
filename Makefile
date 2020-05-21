# your choice of compiler
CC = mpicc

# Add your choice of flags
CFLAGS = -O3 -Wall -Wextra -g
LDLIBS = -lm

all : cg_scatter

cg : cg.o mmio.o
cg_scatter : cg_scatter.o mmio.o
mmio.o : mmio.c mmio.h
cg.o : cg.c mmio.h
cg_scatter.o : cg_scatter.c mmio.h

.PHONY: clean
clean :
	rm -rf *.o cg
