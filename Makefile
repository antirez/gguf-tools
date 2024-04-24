all: gguf-tools

gguf-tools: gguf-tools.c gguflib.c gguflib.h sds.c sds.h sdsalloc.h fp16.h
	$(CC) gguf-tools.c gguflib.c sds.c fp16.c \
		-ffast-math \
		-g3 -Wall -W -pedantic -O3 -o gguf-tools

clean:
	rm -rf gguf-tools
