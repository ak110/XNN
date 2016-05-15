TARGETS = help all XNN test clean valgrind sgcheck demo

help:
	@echo usage:
	@for i in $(TARGETS) ; do echo " make $$i" ; done

all: XNN test

XNN:
	cd XNN && $(MAKE)

test: 
	cd XNNTest && $(MAKE)

clean:
	cd XNN && $(MAKE) clean
	cd XNNTest && $(MAKE) clean

valgrind:
	cd XNNTest && $(MAKE) valgrind

sgcheck:
	cd XNNTest && $(MAKE) sgcheck

demo: XNN
	cd Demo && ./RunAll.sh

.PHONY: $(TARGETS)

