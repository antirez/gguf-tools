all: gguf-tools

gguf-tools: gguf-tools.c gguflib.c gguflib.h sds.c sds.h sdsalloc.h fp16.h bf16.h
	$(CC) gguf-tools.c gguflib.c sds.c fp16.c \
		-march=native -ffast-math \
		-g -ggdb -Wall -W -pedantic -O3 -o gguf-tools

clean:
	rm -rf gguf-tools
