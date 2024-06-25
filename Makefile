
SUPERLU_CFLAGS ?= $(shell pkg-config superlu_dist --cflags )
SUPERLU_LIBS ?= $(shell pkg-config superlu_dist --libs )

CC ?= mpicc
CFLAGS ?= -O0 -g3
LDFLAGS ?= $(shell pkg-config openblas --libs) -fopenmp -lm

pztest: pztest.c
	$(CC) $(CFLAGS) $(SUPERLU_CFLAGS) pztest.c $(SUPERLU_LIBS) -o pztest $(LDFLAGS)

pztest3d: pztest3d.c
	$(CC) $(CFLAGS) $(SUPERLU_CFLAGS) pztest3d.c $(SUPERLU_LIBS) -o pztest3d $(LDFLAGS)

clean:
	rm -f pztest pztest3d
