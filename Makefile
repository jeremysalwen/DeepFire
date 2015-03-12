CXXFLAGS+=-g -O3 -ffast-math -Iinclude -std=c++11 -Wall
LDFLAGS+= -g -O3 -ffast-math -lafopencl 
SRC=src/*.cpp

all: libdeepfire.so test1

libdeepfire.so: $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -shared -fPIC $(SRC) -o libdeepfire.so 
test1: test/test1.o include/*.hpp
	$(CXX) $(LDFLAGS) test/test1.o -o test1

clean:
	rm -f libdeepfire.so test/*.o src/*.o
%.o: %.cxx
	$(CXX) $(CXXFLAGS) %(CPPFLAGS) -c $<
