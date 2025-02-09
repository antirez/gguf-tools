cmake_minimum_required(VERSION 3.5)
project(gguf-tools)
set(SRC_LIST gguf-tools.c gguflib.c sds.c fp16.c)
add_executable(gguf-tools ${SRC_LIST})
