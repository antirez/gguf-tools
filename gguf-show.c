#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>

#include "gguf.h"

typedef struct {
    int fd;
    uint8_t *data;  // Memory mapped data.
    uint64_t size;  // Total file size.
    struct gguf_header *header;     // GUFF file header info.
    uint32_t left_kv;               // Number of key-value pairs yet to read.
    uint32_t left_tensors;          // Number of tensors yet to read.
    uint64_t off;                   // Offset of the next item to parse.
} gguf_ctx;

/* Open a GGUF file and return a parsing context. */
gguf_ctx *gguf_init(char *filename) {
    struct stat sb;
    int fd = open(filename,O_RDONLY);
    if (fd == -1) return NULL;
    if (fstat(fd,&sb) == -1) {
        close(fd);
        return NULL;
    }

    /* Now that we have an open file and its total size, let's
     * mmap it. */
    void *mapped = mmap(0,sb.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if (mapped == MAP_FAILED) {
        close(fd);
        return NULL;
    }

    /* Minimal sanity check... */
    if (sb.st_size < (signed)sizeof(struct gguf_header) ||
        memcmp(mapped,"GGUF",4) != 0)
    {
        errno = EINVAL;
        return NULL;
    }

    /* Mapping successful. We can create our context object. */
    gguf_ctx *ctx = malloc(sizeof(*ctx));
    ctx->fd = fd;
    ctx->data = mapped;
    ctx->header = mapped;
    ctx->size = sb.st_size;
    ctx->off = sizeof(struct gguf_header);
    ctx->left_kv = ctx->header->metadata_kv_count;
    ctx->left_tensors = ctx->header->tensor_count;
    return ctx;
}

/* Cleanup needed after gguf_init(), to terminate the context
 * and cleanup resources. */
void gguf_end(gguf_ctx *ctx) {
    if (ctx == NULL) return;
    munmap(ctx->data,ctx->size);
    close(ctx->fd);
    free(ctx);
}

/* Parse the next key. Returns key information into 'key'.
 * The function return value is 1 is a key was returned, or 0
 * if there are no longer keys to process in this GGUF file. */
int gguf_get_key(gguf_ctx *ctx, gguf_key *key) {
    if (ctx->left_kv == 0) return 0;
    ctx->left_kv--;
    struct gguf_string *str = (struct gguf_string*) (ctx->data+ctx->off);
    key->namelen = str->len;
    key->name = str->string;
    uint32_t *type = (uint32_t*) (ctx->data+ctx->off+8+str->len);
    key->type = *type;
    ctx->off += 8+str->len+4; // Skip prefixed len + string + type.
    key->val = (void*)(ctx->data+ctx->off);
    return 1;
}

/* Return the value type name given the type ID. */
const char *gguf_get_value_type_name(uint32_t type) {
    if (type >= sizeof(gguf_value_name)/sizeof(char*)) return "unknown";
    return gguf_value_name[type];
}

/* Return the length of the value pointed by 'val' of type 'type'.
 * For the array type the length can't be inferred without consuming
 * it, so 0 is returned. */
uint64_t gguf_value_len(uint32_t type, union gguf_value *val) {
    uint64_t valuelen = 0;
    switch(type) {
    case GGUF_VALUE_TYPE_BOOL:
    case GGUF_VALUE_TYPE_UINT8:
    case GGUF_VALUE_TYPE_INT8:
        valuelen = 1; break;
    case GGUF_VALUE_TYPE_UINT16:
    case GGUF_VALUE_TYPE_INT16:
        valuelen = 2; break;
    case GGUF_VALUE_TYPE_UINT32:
    case GGUF_VALUE_TYPE_INT32:
    case GGUF_VALUE_TYPE_FLOAT32:
        valuelen = 4; break;
    case GGUF_VALUE_TYPE_UINT64:
    case GGUF_VALUE_TYPE_INT64:
    case GGUF_VALUE_TYPE_FLOAT64:
        valuelen = 8; break;
    case GGUF_VALUE_TYPE_STRING:
        valuelen = 8+val->string.len; break;
    }
    return valuelen;
}

/* This function can be called after gguf_get_key(), since the context
 * offset will be in the position of a value.
 *
 * The function will process the value, including nested values (in the
 * case of an array value), and for each value will call the specified
 * callback. As a side effect of calling this function, the context offset
 * is advanced to consume the value.
 *
 * If the callback is set to NULL, no callback will be called,
 * but the value will be consumed, so that it will be possible
 * to call gguf_get_key() or gguf_get_tensor() to continue reading
 * the file.
 *
 * When the callback is called, it gets the argument 'privdata' and 'in_array'
 * as passed to this function. This is useful if the callback needs
 * to take state (for pretty printing or alike) and to know if the
 * elements it is processing belong to an array.
 *
 * The value of 'in_array' is the 1-based index of the element being
 * processed.
 *
 * In the case of arrays, callbacks are also called with the special
 * type ARRAY_START / ARRAY_END at the start/end of the array
 * processing. */
void gguf_do_with_value(gguf_ctx *ctx, uint32_t type, union gguf_value *val,
                        void *privdata, uint64_t in_array, uint64_t array_len,
                        void(*callback)(void *privdata, uint32_t type,
                                     union gguf_value *val, uint64_t in_array,
                                     uint64_t array_len))
{
    if (type == GGUF_VALUE_TYPE_ARRAY) {
        uint32_t etype; // Elements type.
        uint64_t len;   // Number of elements.
        etype = val->array.type;
        len = val->array.len;
        //exit(1);
        ctx->off += 4+8; // Skip elements type / array length.
        callback(privdata,GGUF_VALUE_TYPE_ARRAY_START,val,in_array,len);
        for (uint64_t j = 0; j < len; j++) {
            val = (union gguf_value*)(ctx->data+ctx->off);
            gguf_do_with_value(ctx,etype,val,privdata,j+1,len,callback);
            /* As a side effect of calling gguf_do_with_value() ctx->off
             * will be update, so 'val' will be set to the next element. */
        }
        callback(privdata,GGUF_VALUE_TYPE_ARRAY_END,NULL,in_array,len);
    } else {
        callback(privdata,type,val,in_array,array_len);
        ctx->off += gguf_value_len(type,val);
    }
}

struct gguf_print_options {
    uint64_t max_array_items;       // Don't print more than N items.
};

/* Print a GGUF value. 'privdata' is used to pass guff_print_options and
 * may be NULL if no options are provided.
 *
 * The function is designed to be used as a callback of gguf_do_with_value(). */
void gguf_print_value_callback(void *privdata, uint32_t type, union gguf_value *val, uint64_t in_array, uint64_t array_len) {
    (void) privdata;

    switch (type) {
        case GGUF_VALUE_TYPE_ARRAY_START:
            printf("[(%llu items)",array_len); break;
        case GGUF_VALUE_TYPE_ARRAY_END:
            printf("]"); break;
        case GGUF_VALUE_TYPE_UINT8:
            printf("%u", val->uint8); break;
        case GGUF_VALUE_TYPE_INT8:
            printf("%d", val->int8); break;
        case GGUF_VALUE_TYPE_UINT16:
            printf("%u", val->uint16); break;
        case GGUF_VALUE_TYPE_INT16:
            printf("%d", val->int16); break;
        case GGUF_VALUE_TYPE_UINT32:
            printf("%u", val->uint32); break;
        case GGUF_VALUE_TYPE_INT32:
            printf("%d", val->int32); break;
        case GGUF_VALUE_TYPE_FLOAT32:
            printf("%f", val->float32); break;
        case GGUF_VALUE_TYPE_BOOL:
            if (val->boolval == 0 || val->boolval == 1)
                printf("%s", val->boolval ? "true" : "false");
            else
                printf("Invalid boolean value %d", val->boolval);
            break;
        case GGUF_VALUE_TYPE_STRING:
            printf("%.*s", (int)val->string.len, val->string.string); break;
        case GGUF_VALUE_TYPE_UINT64:
            printf("%llu", val->uint64); break;
        case GGUF_VALUE_TYPE_INT64:
            printf("%lld", val->int64); break;
        case GGUF_VALUE_TYPE_FLOAT64:
            printf("%lf", val->float64); break;
        default:
            printf("Unknown type\n");
            break;
    }
    if (in_array && in_array != array_len) printf(", ");
}

/* Print the current value, including arrays. As a side effect
 * the value will be consumed from the context, that will now point
 * to the next item in the GGUF file. */
void gguf_print_value(gguf_ctx *ctx, uint32_t type, union gguf_value *val) {
    gguf_do_with_value(ctx,type,val,NULL,0,0,gguf_print_value_callback);
}

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
        gguf_print_value(ctx,key.type,key.val);
        printf("\n");
    }
    return 0;
}
