SRCS     = $(wildcard *.cpp)
OBJS     = $(SRCS:.cpp=.o)
CXXFLAGS = -Wall -g -O3 -march=native -std=c++11 -fopenmp
CPPFLAGS = -DNDEBUG
LDFLAGS  = -pthread -lgomp -fopenmp

all: XNN

clean:
	rm -f XNN $(OBJS)

.PHONY: all clean

XNN: $(OBJS)
	$(CXX) $(LDFLAGS) $(TARGET_ARCH) -o $@ $^

$(OBJS): $(SRCS) $(wildcard *.h)

