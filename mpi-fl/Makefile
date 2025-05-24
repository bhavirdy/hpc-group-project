# Makefile for Federated K-Means Clustering
CXX = mpicxx
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra
TARGET = federated_kmeans
SOURCES = federated_kmeans.cpp

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Run on cluster
cluster-run: $(TARGET)
	@echo "Running on cluster with hostfile..."
	mpirun -np 4 -hostfile hostfile ./$(TARGET) data.csv 3

.PHONY: all clean install-deps test cluster-run generate-data