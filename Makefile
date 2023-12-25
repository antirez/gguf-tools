all: gguf-tools

gguf-tools: gguf-tools.c gguflib.c gguflib.h sds.c sds.h sdsalloc.h
	$(CC) gguf-tools.c gguflib.c sds.c \
		-g -ggdb -Wall -W -pedantic -O2 -o gguf-tools

clean:
	rm -rf gguf-tools
