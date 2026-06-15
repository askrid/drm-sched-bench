CC ?= cc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lvulkan -lm

all: vkload spin.spv

vkload: vkload.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

spin.spv: spin.comp
	glslangValidator -V $< -o $@

clean:
	rm -f vkload spin.spv

.PHONY: all clean
