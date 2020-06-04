# your choice of compiler
CC = mpicc

# Add your choice of flags
CFLAGS = -O3 -Wall -Wextra -g
LDLIBS = -lm

all :clean cg_scatter cg cg_a

cg : cg.o mmio.o
cg_scatter : cg_scatter.o mmio.o
cg_a : cg_a.o mmio.o
mmio.o : mmio.c mmio.h
cg.o : cg.c mmio.h
cg_scatter.o : cg_scatter.c mmio.h
cg_a.o : cg_a.c mmio.h
.PHONY: clean
clean :
	rm -rf *.o cg cg_scatter cg_a
