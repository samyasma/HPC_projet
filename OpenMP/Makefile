
# your choice of compiler
CC = gcc

# Add your choice of flags
CFLAGS = -O3 -Wall -Wextra -g -fopenmp
LDLIBS = -lm -fopenmp

all : clean cg cg_openmp

cg : cg.o mmio.o
cg_openmp : cg_openmp.o mmio.o
mmio.o : mmio.c mmio.h
cg.o : cg.c mmio.h
cg_openmp.o : cg_openmp.c mmio.h

.PHONY: clean
clean :
	rm -rf *.o cg cg_openmp
