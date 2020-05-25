# your choice of compiler
CC = gcc-9

# Add your choice of flags
CFLAGS = -fopenmp
LDFLAGS = -fopenmp
LDLIBS = -lm

all : clean cg

cg : cg.o mmio.o
mmio.o : mmio.c mmio.h
cg.o : cg.c mmio.h

.PHONY: clean
clean :
	rm -rf *.o cg
