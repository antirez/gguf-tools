all: gguf-show

gguf-show: gguf-show.c gguflib.c gguflib.h
	$(CC) gguf-show.c gguflib.c -g -ggdb -Wall -W -pedantic -O2 -o gguf-show

clean:
	rm -rf gguf-show
