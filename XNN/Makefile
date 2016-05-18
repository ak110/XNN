SRCS     = $(wildcard *.cpp)
OBJS     = $(SRCS:.cpp=.o)
CXXFLAGS = -Wall -g -O3 -march=native -std=c++14 -fopenmp
CPPFLAGS = -DNDEBUG
LDFLAGS  = -pthread
ifeq ($(USE_LIBIOMP), 1)
  LDFLAGS += -liomp5
else
  LDFLAGS += -lgomp -fopenmp
endif

all: XNN

clean:
	rm -f XNN $(OBJS)

.PHONY: all clean

XNN: $(OBJS)
	$(CXX) $(LDFLAGS) $(TARGET_ARCH) -o $@ $^

$(OBJS): $(SRCS) $(wildcard *.h)

