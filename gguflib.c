#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

#include "gguflib.h"

/* ============================ Low level functions ========================= */

/* GGUF value ID to name lookup table. */
const char *gguf_value_name[] = {
    "uint8", "int8", "uint16", "int16", "uint32", "int32",
    "float32", "bool", "string", "array", "uint64", "int64",
    "float64"
};

/* GGUF tensor type to features lookup table. */
struct gguf_tensor_type_features {
    char *name;
    uint32_t items_per_block;
    uint32_t bytes_per_block;
} gguf_tensor_type_features[] = {
    {"f32", 1, 4},
    {"f16", 1, 2},
    {"q4_0", 32, 18},
    {"q4_1", 32, 20},
    {"q4_2 deprecated", 0, 0},
    {"q4_3 deprecated", 0, 0},
    {"q5_0", 32, 22},
    {"q5_1", 32, 24},
    {"q8_0", 32, 34},
    {"q8_1", 32, 40},
    {"q2_k", 256, 82},
    {"q3_k", 256, 110},
    {"q4_k", 256, 144},
    {"q5_k", 256, 176},
    {"q6_k", 256, 210},
    {"q8_k", 256, 292},
};

/* Return the value type name given the type ID. */
const char *gguf_get_value_type_name(uint32_t type) {
    if (type >= sizeof(gguf_value_name)/sizeof(char*)) return "unknown";
    return gguf_value_name[type];
}

/* Return the tensor type name given the type ID. */
const char *gguf_get_tensor_type_name(uint32_t type) {
    if (type >= sizeof(gguf_tensor_type_features)/sizeof(gguf_tensor_type_features[0])) return "unknown";
    return gguf_tensor_type_features[type].name;
}

/* Return the tensor type features, or NULL if the type ID is out of range. */
struct gguf_tensor_type_features *gguf_get_tensor_type_features(uint32_t type) {
    if (type >= sizeof(gguf_tensor_type_features)/sizeof(gguf_tensor_type_features[0])) return NULL;
    return &gguf_tensor_type_features[type];
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

/* =============================== GGUF file API ============================ */

/* Open a GGUF file and return a parsing context. */
gguf_ctx *gguf_init(const char *filename) {
    int fd = open(filename,O_RDWR|O_APPEND);
    if (fd == -1) return NULL;

    /* Mapping successful. We can create our context object. */
    gguf_ctx *ctx = malloc(sizeof(*ctx));
    memset(ctx,0,sizeof(*ctx));
    ctx->fd = fd;
    ctx->alignment = 32; // Default alighment of GGUF files.
    ctx->data_off = 0;   // Set later.
    if (gguf_remap(ctx) == 0) {
        gguf_end(ctx);
        return NULL;
    }
    gguf_rewind(ctx);
    return ctx;
}

/* Set the context to read the first key-value entry in the GGUF
 * file and then all the rest. Is used when creating a new context
 * and also when you want to restart scanning the key-value
 * items in the file. */
void gguf_rewind(gguf_ctx *ctx) {
    ctx->off = sizeof(struct gguf_header);
    ctx->left_kv = ctx->header->metadata_kv_count;
    ctx->left_tensors = ctx->header->tensor_count;
}

/* map or re-map the GGUF file inside the context pointers to
 * header and data, also calculating the file length. This is
 * used when creating a context, but also after the user write
 * to the file extending it, and requires to view again the
 * whole updated file.
 *
 * Return 1 on success, 0 on error. */
int gguf_remap(gguf_ctx *ctx) {
    struct stat sb;

    /* Unmap if the file was already memory mapped. */
    if (ctx->data) munmap(ctx->data,ctx->size);

    /* Get the size of the file to map, then map it. */
    if (fstat(ctx->fd,&sb) == -1) return 0;

    void *mapped = mmap(0,sb.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,ctx->fd,0);
    if (mapped == MAP_FAILED) return 0;

    /* Minimal sanity check... */
    if (sb.st_size < (signed)sizeof(struct gguf_header) ||
        memcmp(mapped,"GGUF",4) != 0)
    {
        errno = EINVAL;
        return 0;
    }
    ctx->data = mapped;
    ctx->header = mapped;
    ctx->size = sb.st_size;
    return 1;
}

/* Cleanup needed after gguf_init(), to terminate the context
 * and cleanup resources. */
void gguf_end(gguf_ctx *ctx) {
    if (ctx == NULL) return;
    if (ctx->data) munmap(ctx->data,ctx->size);
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

    /* Update the context with the alignmnet data, if needed. */
    const char *alignment_key = "general.alignmnet";
    if (key->type == GGUF_VALUE_TYPE_UINT32 &&
        key->namelen == strlen(alignment_key) &&
        memcmp(alignment_key, key->name, key->namelen) == 0)
    {
        ctx->alignment = key->val->uint32;
    }
    return 1;
}

/* Given an offset or a length, returns the padding needed to align it
 * to ctx->alignment. */
uint64_t gguf_get_alignment_padding(uint64_t alignment, uint64_t offset) {
    return (alignment - (offset % alignment)) % alignment;
}

/* Set the data section offset. This function must be called exactly when
 * all the key-values are consumed, in the context of the first call of
 * gguf_get_tensor(): this way we will be able to return tensor offsets
 * as absolute positions and pointers to the mmapped file. */
void gguf_set_data_offset(gguf_ctx *ctx) {
    assert(ctx->left_kv == 0 && ctx->left_tensors == ctx->header->tensor_count);

    uint64_t offset = ctx->off;
    for (uint32_t j = 0; j < ctx->left_tensors; j++) {
        struct gguf_string *str = (struct gguf_string*) (ctx->data+offset);
        offset += 8+str->len;   // Skip prefixed len + string
        uint32_t *num_dim = (uint32_t*)(ctx->data+offset);
        offset += 4;            // Skip num dimentions.
        offset += 8*(*num_dim); // Skip dimensions.
        offset += 4;            // Skip tensor type.
        offset += 8;            // Skip tensor offset.
    }
    uint64_t padding = gguf_get_alignment_padding(ctx->alignment,offset);
    ctx->data_off = offset + padding;
}

/* Parse the next tensor info data. Returns information into 'tensor'.
 * The function return value is 1 is a tensor was returned, or 0
 * if there are no longer tensors to process in this GGUF file or if
 * there are still key-value pairs to process before getting into the
 * tensors section.
 *
 * When 0 is returned, we are at the end of the file and as a side
 * effect this function will set the data offset ctx->data_off. */
int gguf_get_tensor(gguf_ctx *ctx, gguf_tensor *tensor) {
    if (ctx->left_tensors == 0 || ctx->left_kv != 0) return 0;

    /* We want to return tensor data with offsets relative to the start
     * of the file, so that the user of the API is able to access tensors
     * as it iterates over them. To do so, we need to perform a fulls
     * scan if this is the first tensor info we are reading. */
    if (ctx->data_off == 0) gguf_set_data_offset(ctx);

    ctx->left_tensors--;
    struct gguf_string *str = (struct gguf_string*) (ctx->data+ctx->off);
    ctx->off += 8+str->len; // Skip prefixed len + string.
    tensor->namelen = str->len;
    tensor->name = str->string;
    uint32_t *num_dim = (uint32_t*) (ctx->data+ctx->off);
    ctx->off += 4;  // Skip number of dimensions.
    tensor->ndim = *num_dim;
    assert(tensor->ndim <= GGUF_TENSOR_MAX_DIM);

    /* Read the dimentions: all the unused dimentions are set to 1. */
    tensor->num_weights = 1;
    for (uint32_t j = 0; j < tensor->ndim; j++) {
        if (j < tensor->ndim) {
            uint64_t *dim = (uint64_t*) (ctx->data+ctx->off);
            ctx->off += 8; // Skip dimension size.
            tensor->dim[j] = *dim;
            tensor->num_weights *= *dim;
        } else {
            tensor->dim[j] = 1;
        }
    }
    uint32_t *type = (uint32_t*) (ctx->data+ctx->off);
    ctx->off += 4;  // Skip tensor type.
    tensor->type = *type;

    uint64_t *offset = (uint64_t*) (ctx->data+ctx->off);
    ctx->off += 8;  // Skip tensor offset.

    tensor->offset = ctx->data_off + *offset;
    tensor->weights_data = ctx->data + tensor->offset;

    struct gguf_tensor_type_features *tf;
    tf = gguf_get_tensor_type_features(tensor->type);
    uint64_t weights_padding = gguf_get_alignment_padding(tf->items_per_block,tensor->num_weights);
    tensor->bsize = ((tensor->num_weights+weights_padding) / tf->items_per_block) * tf->bytes_per_block;
    return 1;
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
        if (callback)
            callback(privdata,GGUF_VALUE_TYPE_ARRAY_START,val,in_array,len);
        for (uint64_t j = 0; j < len; j++) {
            val = (union gguf_value*)(ctx->data+ctx->off);
            gguf_do_with_value(ctx,etype,val,privdata,j+1,len,callback);
            /* As a side effect of calling gguf_do_with_value() ctx->off
             * will be update, so 'val' will be set to the next element. */
        }
        if (callback)
            callback(privdata,GGUF_VALUE_TYPE_ARRAY_END,NULL,in_array,len);
    } else {
        if (callback)
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
    struct gguf_print_options *po = privdata;
    if (po && po->max_array_items && in_array > po->max_array_items) {
        if (in_array-1 == po->max_array_items)
            printf("... %llu more items of %llu", array_len-in_array+1,
                                                  array_len);
        return;
    }

    switch (type) {
        case GGUF_VALUE_TYPE_ARRAY_START:
            printf("["); break;
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
 * to the next item in the GGUF file.
 *
 * If 'full' is true, in the case of arrays, the whole array is printed,
 * otherwise just the first few elements. */
void gguf_print_value(gguf_ctx *ctx, uint32_t type, union gguf_value *val, int full) {
    struct gguf_print_options po;
    po.max_array_items = full ? 0 : 30;
    gguf_do_with_value(ctx,type,val,&po,0,0,gguf_print_value_callback);
}

/* ============================= GGUF writing API  ========================== */

/* Create an empty GGUF file with no key-value pairs nor tensors.
 * The file can be extended by using the APIs to add tensors and
 * keys.
 *
 * On success the context with the file already loaded is returned,
 * otherwise NULL is returned. */
gguf_ctx *gguf_create(const char *filename) {
    struct gguf_header hdr;
    memcpy(&hdr.magic,"GGUF",4);
    hdr.version = 3;
    hdr.tensor_count = 0;
    hdr.metadata_kv_count = 0;

    FILE *fp = fopen(filename,"wx");
    if (fp == NULL) return NULL;
    if (fwrite(&hdr,1,sizeof(hdr),fp) != sizeof(hdr)) {
        fclose(fp);
        return NULL;
    }
    fclose(fp);

    return gguf_init(filename);
}

/* Low level API to append some key-value data to the GGUF file identified
 * by the context 'ctx'. It's up to the caller to provide a well-formatted
 * value of the specified type in 'val'. The len is the raw bytes length of
 * the specified value. Higher level APIs use this one to create fields with
 * different numerical values, strings, ...
 *
 * On success the function returns 1. Otherwise 0.
 * The function fails and returns 0 with errno set to EINVAL if the
 * tensors count in the header is non-zero: we can't append key-value
 * data after the first tensor was emitted. */
int gguf_append_kv(gguf_ctx *ctx, const char *keyname, uint64_t keylen, uint32_t type, void *val, uint64_t len) {
    if (ctx->header->tensor_count != 0) {
        errno = EINVAL;
        return 0;
    }
    if (write(ctx->fd,&keylen,sizeof(keylen)) != sizeof(keylen)) return 0;
    if (write(ctx->fd,keyname,keylen) != (ssize_t)keylen) return 0;
    if (write(ctx->fd,&type,sizeof(type)) != sizeof(type)) return 0;
    if (write(ctx->fd,val,len) != (ssize_t)len) return 0;
    gguf_remap(ctx);
    ctx->header->metadata_kv_count++;
    return 1;
}

/* Append tensor metadata (but not the actual tensor weights data) to the
 * GGUF file identified by 'ctx'. */
int gguf_append_tensor(gguf_ctx *ctx, const char *tensorname, uint64_t namelen, uint32_t num_dim, uint64_t *dim, uint32_t type, uint64_t offset)
{
    if (write(ctx->fd,&namelen,sizeof(namelen)) != sizeof(namelen)) return 0;
    if (write(ctx->fd,tensorname,namelen) != (ssize_t)namelen) return 0;
    if (write(ctx->fd,&num_dim,sizeof(num_dim)) != sizeof(num_dim)) return 0;
    for (uint32_t j = 0; j < num_dim; j++) {
        if (write(ctx->fd,&dim[j],sizeof(uint64_t)) != sizeof(uint64_t))
            return 0;
    }
    if (write(ctx->fd,&type,sizeof(type)) != sizeof(type)) return 0;
    if (write(ctx->fd,&offset,sizeof(offset)) != sizeof(offset)) return 0;
    gguf_remap(ctx);
    ctx->header->tensor_count++;
    return 1;
}

/* Append tensor data enforcing the GGUF file aligment. The user must specify
 * an offset that requires no more than ctx.alignemnt-1 padding bytes starting
 * from the current offset (this means that this function should be called
 * sequentially for all the tensors we want to store, after we already
 * computed the right offset for all the tensors). Also the offset must be
 * aligned. Otherwise the function will fail returning 0. On success, 1 is
 * returned. The function will take care to add the padding required to
 * start writing the tensor at the specified offset. */
int gguf_append_tensor_data(gguf_ctx *ctx, uint64_t offset, void *tensor, uint64_t tensor_size) {
    char padding_data[1024] = {0};
    assert(sizeof(padding_data) >= ctx->alignment);

    /* Is the offset aligned? */
    if (offset % ctx->alignment) {
        errno = EINVAL;
        return 0;
    }

    /* We expect the offset of the context to be already where this tensor
     * should be stored, minus the padding. */
    if (offset < ctx->off || offset - ctx->off >= ctx->alignment) {
        errno = EINVAL;
        return 0;
    }

    uint64_t padding = gguf_get_alignment_padding(ctx->alignment,offset);
    if (write(ctx->fd,padding_data,padding) != (ssize_t)padding) return 0;
    if (write(ctx->fd,tensor,tensor_size) != (ssize_t)tensor_size) return 0;
    gguf_remap(ctx);
    return 1;
}
