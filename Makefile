
SUPERLU_CFLAGS ?= $(shell pkg-config superlu_dist --cflags )
SUPERLU_LIBS ?= $(shell pkg-config superlu_dist --libs )

CXX ?= mpicxx
NVCC ?= nvcc
CXXFLAGS ?= -O0 -g3 
LDFLAGS ?= $(shell pkg-config openblas --libs) -fopenmp -lm
NVCCFLAGS ?= -g -O0
CUSPARSE_LIBS ?= -lcusparse -lcublas -lcudart

pztest: pztest.c
	$(CC) $(CFLAGS) $(SUPERLU_CFLAGS) pztest.c $(SUPERLU_LIBS) -o pztest $(LDFLAGS)

pztest3d: pztest3d.cpp
	$(CXX) $(CXXFLAGS) $(SUPERLU_CFLAGS) pztest3d.cpp $(SUPERLU_LIBS) -o pztest3d $(LDFLAGS)

pztest3d_cusparse: pztest3d_cusparse.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler '$(CXXFLAGS)'  pztest3d_cusparse.cpp -o pztest3d_cusparse $(CUSPARSE_LIBS)

clean:
	rm -f pztest pztest3d
