CXX = g++
CXXFLAGS = -O3 -std=c++17 
OMPFLAGS = -fopenmp
 
all: qlearning_serial qlearning_shared qlearning_localsync
 
qlearning_serial: qlearning_serial.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
 
qlearning_shared: qlearning_shared.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@
 
qlearning_localsync: qlearning_localsync.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@
 
ifeq ($(OS),Windows_NT)
clean:
	del /Q qlearning_serial.exe qlearning_shared.exe qlearning_localsync.exe
else
clean:
	rm -f qlearning_serial qlearning_shared qlearning_localsync
endif
 
.PHONY: all clean