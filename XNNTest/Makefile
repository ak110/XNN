SRCS     = $(wildcard *.cpp)
OBJS     = $(SRCS:.cpp=.o) gtest/src/gtest-all.o
CXXFLAGS = -Wall -g -O1 -march=native -std=c++14 -fopenmp
CPPFLAGS = -Igtest -Igtest/include
LDFLAGS  = -pthread -lgomp -fopenmp

test: XNNTest
	cd ../XNN && "$(CURDIR)/XNNTest"

valgrind: XNNTest
	cd ../XNN && echo "ulimit -c unlimited ; valgrind --tool=memcheck --leak-check=full --track-origins=yes --verbose --log-file=valgrind.log $(CURDIR)/XNNTest" | $(SHELL)

sgcheck: XNNTest
	cd ../XNN && echo "ulimit -c unlimited ; valgrind --tool=exp-sgcheck --verbose --log-file=valgrind-sgcheck.log $(CURDIR)/XNNTest" | $(SHELL)

clean:
	rm -f XNNTest $(OBJS)

.PHONY: test valgrind sgcheck clean

XNNTest: $(OBJS)
	$(CXX) $(LDFLAGS) $(TARGET_ARCH) -o $@ $^

$(OBJS):  ../XNN/*.cpp

