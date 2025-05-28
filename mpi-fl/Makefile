# Makefile for Federated K-Means Clustering
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
TARGET = federated_kmeans
SOURCES = src/federated_kmeans.cpp

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

.PHONY: all clean