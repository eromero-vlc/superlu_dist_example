
SUPERLU_CFLAGS ?= $(shell pkg-config superlu_dist --cflags )
SUPERLU_LIBS ?= $(shell pkg-config superlu_dist --libs )

CXX ?= mpicxx
CXXFLAGS ?= -O0 -g3 
LDFLAGS ?= $(shell pkg-config openblas --libs) -fopenmp -lm

pztest: pztest.c
	$(CC) $(CFLAGS) $(SUPERLU_CFLAGS) pztest.c $(SUPERLU_LIBS) -o pztest $(LDFLAGS)

pztest3d: pztest3d.cpp
	$(CXX) $(CXXFLAGS) $(SUPERLU_CFLAGS) pztest3d.cpp $(SUPERLU_LIBS) -o pztest3d $(LDFLAGS)

clean:
	rm -f pztest pztest3d
