# Compiler settings
MPICXX = mpicxx
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra

# Targets
FED_TARGET = fed_kmeans
CENT_TARGET = cent_kmeans

# Sources
FED_SRC = src/fed_kmeans.cpp
CENT_SRC = src/cent_kmeans.cpp

all: $(FED_TARGET) $(CENT_TARGET)

$(FED_TARGET): $(FED_SRC)
	$(MPICXX) $(CXXFLAGS) -o $@ $^

$(CENT_TARGET): $(CENT_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(FED_TARGET) $(CENT_TARGET) *.o
