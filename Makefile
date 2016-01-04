CXXFLAGS+=-g -O3 -ffast-math -fPIC -Iinclude -std=c++11 -Wall
LDFLAGS+= -g -O3 -ffast-math -lafopencl -std=c++11 -Wall
SRC=$(wildcard src/*.cpp)
OBJS=$(patsubst %.cpp, %.o, $(SRC))

all: libdeepfire.so test1

libdeepfire.so: $(OBJS)
	$(CXX) $(LDFLAGS) -shared -fPIC $(OBJS) -o libdeepfire.so 

test1: test/test1.o include/*.hpp
	$(CXX)  $(LDFLAGS) -L. -ldeepfire test/test1.o -o test1

clean:
	rm -f libdeepfire.so test/*.o src/*.o
%.o: %.cxx
	$(CXX) $(CXXFLAGS) %(CPPFLAGS) -c $<
