TARGETS = help all XNN test clean valgrind sgcheck demo1 demo2 demo3 demo4 demo-all

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

demo1: XNN
	cd Demo/01_BinaryClassification1 && ./Run.sh

demo2: XNN
	cd Demo/02_BinaryClassification2 && ./Run.sh

demo3: XNN
	cd Demo/03_Regression && ./Run.sh

demo4: XNN
	cd Demo/04_MulticlassClassification && ./Run.sh

demo-all: XNN
	cd Demo && ./RunAll.sh

.PHONY: $(TARGETS)

