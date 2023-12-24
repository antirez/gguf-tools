#include <stdio.h>
#include <stdlib.h>
#include "gguflib.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n",argv[0]);
        exit(1);
    }
    gguf_ctx *ctx = gguf_init(argv[1]);
    if (ctx == NULL) {
        perror("Opening GGUF file");
        exit(1);
    }

    /* Show general information about the neural network. */
    printf("%s (ver %d): %llu key-value pairs, %llu tensors\n",
        argv[1],
        (int)ctx->header->version,
        (unsigned long long)ctx->header->metadata_kv_count,
        (unsigned long long)ctx->header->tensor_count);

    /* Show all the key-value pairs. */
    gguf_key key;
    while (gguf_get_key(ctx,&key)) {
        printf("%.*s: [%s] ", (int)key.namelen, key.name, gguf_get_value_type_name(key.type));
        gguf_print_value(ctx,key.type,key.val,0);
        printf("\n");
    }

    gguf_tensor tensor;
    while (gguf_get_tensor(ctx,&tensor)) {
        printf("%s tensor %.*s @%llu, %llu weights, %llu bytes\n",
            gguf_get_tensor_type_name(tensor.type),
            (int)tensor.namelen,
            tensor.name,
            tensor.offset,
            tensor.num_weights,
            tensor.bsize);
    }
    return 0;
}
