# your choice of compiler
CC = mpicc

# Add your choice of flags
CFLAGS = -O3 -Wall -Wextra -g -fopenmp
LDLIBS = -lm -fopenmp
USE_OMP = -DUSING_OMP

all :clean cg_openmp_mpi

cg : cg.o mmio.o
cg_openmp_mpi :cg_openmp_mpi.o mmio.o
mmio.o : mmio.c mmio.h
cg.o : cg.c mmio.h
cg_openmp_mpi.o : cg_openmp_mpi.c mmio.h

.PHONY: clean
clean :
	rm -rf *.o cg_openmp_mpi
