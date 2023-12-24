all: gguf-show

gguf-show: gguf-show.c gguf.h
	$(CC) gguf-show.c -g -ggdb -Wall -W -pedantic -O2 -o gguf-show

clean:
	rm -rf gguf-show
