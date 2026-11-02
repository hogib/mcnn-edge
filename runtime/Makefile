# sonar_edge C runtime. Dependency-free C11.
#   make            build libsonar_edge.a and the sonar-detect CLI
#   make test       bit-exact golden test against the Python integer simulation
#   make CC=aarch64-linux-gnu-gcc   cross-compile for a 64-bit Raspberry Pi
CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Werror -pedantic
LDFLAGS ?=

all: libsonar_edge.a sonar-detect

sonar_edge.o: sonar_edge.c sonar_edge.h weights.h
	$(CC) $(CFLAGS) -c -o $@ $<

libsonar_edge.a: sonar_edge.o
	$(AR) rcs $@ $^

sonar-detect: cli/sonar_detect.c libsonar_edge.a
	$(CC) $(CFLAGS) -o $@ $< libsonar_edge.a $(LDFLAGS)

tests/test_golden: tests/test_golden.c tests/golden.h libsonar_edge.a
	$(CC) $(CFLAGS) -o $@ $< libsonar_edge.a

test: tests/test_golden
	./tests/test_golden

clean:
	rm -f *.o libsonar_edge.a sonar-detect tests/test_golden

.PHONY: all test clean
