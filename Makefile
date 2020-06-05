# your choice of compiler
CC = mpicc

# Add your choice of flags
CFLAGS = -O3 -Wall -Wextra -g -fopenmp
LDLIBS = -lm -fopenmp
USE_OMP = -DUSING_OMP

all :clean Hybride/cg_openmp_mpi MPI/cg_scatter OpenMP/cg_openmp sequential/cg

Hybride/cg_openmp_mpi : Hybride/cg_openmp_mpi.o Hybride/mmio.o
Hybride/mmio.o : Hybride/mmio.c Hybride/mmio.h
Hybride/cg_openmp_mpi.o : Hybride/cg_openmp_mpi.c Hybride/mmio.h

MPI/cg_scatter : MPI/cg_scatter.o MPI/mmio.o
MPI/mmio.o : MPI/mmio.c MPI/mmio.h
MPI/cg_scatter.o : MPI/cg_scatter.c MPI/mmio.h

OpenMP/cg_openmp : OpenMP/cg_openmp.o OpenMP/mmio.o
OpenMP/mmio.o : OpenMP/mmio.c OpenMP/mmio.h
OpenMP/cg_openmp.o : OpenMP/cg_openmp.c OpenMP/mmio.h

sequential/cg : sequential/cg.o sequential/mmio.o
sequential/mmio.o : sequential/mmio.c sequential/mmio.h
sequential/cg.o : sequential/cg.c sequential/mmio.h


.PHONY: clean
clean :
	rm -rf Hybride/*.o  MPI/*.o OpenMP/*.o sequential/*.o Hybride/cg_openmp_mpi MPI/cg_scatter OpenMP/cg_openmp sequential/cg
