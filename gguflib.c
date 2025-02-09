#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#endif
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>

#ifdef _WIN32
#include <Windows.h>

typedef UINT_PTR ssize_t;
#endif

#include "gguflib.h"
#include "fp16.h"
#include "bf16.h"

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
    {"iq2_xxs", 256, 66},
    {"iq2_xs", 256, 74},
    {"iq3_xxs", 256, 98},
    {"iq1_s", 256, 110},
    {"iq4_nl", 256, 50},
    {"iq3_s", 256, 110},
    {"iq2_s", 256, 82},
    {"iq4_xs", 256, 136},
    {"i8", 1, 1},
    {"i16", 1, 2},
    {"i32", 1, 4},
    {"i64", 1, 8},
    {"f64", 1, 8},
    {"iq1_m", 256, 56},
    {"bf16", 1, 2},
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
gguf_ctx*gguf_open(const char *filename) {
#ifdef _WIN32
    HANDLE fd = CreateFileA(filename, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fd == INVALID_HANDLE_VALUE) return NULL;
#else
    int fd = open(filename, O_RDWR | O_APPEND);
    if (fd == -1) return NULL;
#endif

    /* Mapping successful. We can create our context object. */
    gguf_ctx*ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->fd = fd;
    ctx->alignment = 32; // Default alignment of GGUF files.
    ctx->data_off = 0;   // Set later.

#ifdef _WIN32
    /* We must create file mapping object under Windows. */
    HANDLE mapping = CreateFileMappingA(fd, NULL, PAGE_READWRITE, 0, 0, NULL);
    if (mapping == NULL) {
        CloseHandle(fd);
        free(ctx);
        return 0;
    }
    ctx->mapping = mapping;
#endif

    if (gguf_remap(ctx) == 0) {
        gguf_close(ctx);
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
#ifndef _WIN32
    struct stat sb;

    /* Unmap if the file was already memory mapped. */
    if (ctx->data) munmap(ctx->data,ctx->size);

    /* Get the size of the file to map, then map it. */
    if (fstat(ctx->fd,&sb) == -1) return 0;

    void *mapped = mmap(0,sb.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,ctx->fd,0);
    if (mapped == MAP_FAILED) return 0;

    /* Minimal sanity check... */
    if (sb.st_size < (signed)sizeof(struct gguf_header) ||
        memcmp(mapped, "GGUF", 4) != 0)
    {
        errno = EINVAL;
        return 0;
    }
    ctx->size = sb.st_size;
#else
    if (ctx->data) UnmapViewOfFile(ctx->data);

    /* Get the size of the file. */
    LARGE_INTEGER size;
    if (!GetFileSizeEx(ctx->fd, &size)) return 0;

    /* Map the file by the handle to the file mapping object. */
    LPVOID mapped = MapViewOfFile(ctx->mapping, FILE_MAP_ALL_ACCESS, 0, 0, size.QuadPart);
    if (mapped == NULL) return 0;

    if (size.QuadPart < (signed)sizeof(struct gguf_header) ||
        memcmp(mapped, "GGUF", 4) != 0)
    {
        errno = EINVAL;
        return 0;
    }
    ctx->size = size.QuadPart;
#endif

    ctx->data = mapped;
    ctx->header = mapped;
    return 1;
}

/* Cleanup needed after gguf_open() and gguf_create(), to terminate the context
 * and cleanup resources. */
void gguf_close(gguf_ctx *ctx) {
    if (ctx == NULL) return;
#ifndef _WIN32
    if (ctx->data) munmap(ctx->data,ctx->size);
    close(ctx->fd);
#else
    if (ctx->data) UnmapViewOfFile(ctx->data);
    /* Don't forget to close the handle to the file mapping object to destory this kernel object. */
    CloseHandle(ctx->mapping);
    CloseHandle(ctx->fd);
#endif
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

    /* Update the context with the alignment data, if needed. */
    const char *alignment_key = "general.alignment";
    if (key->type == GGUF_VALUE_TYPE_UINT32 &&
        key->namelen == strlen(alignment_key) &&
        memcmp(alignment_key, key->name, key->namelen) == 0)
    {
        ctx->alignment = key->val->uint32;
    }
    return 1;
}

/* Skip all the key values pairs in the GGUF files to get to the
 * tensors information segment. */
void gguf_skip_key_values_section(gguf_ctx *ctx) {
    gguf_key key;
    while (gguf_get_key(ctx,&key))
        gguf_do_with_value(ctx,key.type,key.val,NULL,0,0,NULL);
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
static void gguf_set_data_offset(gguf_ctx *ctx) {
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
 * The function return value is 1 if a tensor was returned, or 0
 * if there are no longer tensors to process in this GGUF file or if
 * there are still key-value pairs to process before getting into the
 * tensors section.
 *
 * The first time this function is called, as a side effect it will
 * set ctx->data_off to return tensors with absolute offsets.
 * 
 * When 0 is returned, the tensor name is set to NULL, so that after
 * a while() loop scanning tensors for a given condition, the caller
 * can easily understand if the search terminated because the loop
 * was exit or because all the entries were consumed. */
int gguf_get_tensor(gguf_ctx *ctx, gguf_tensor *tensor) {
    if (ctx->left_tensors == 0 || ctx->left_kv != 0) {
        tensor->name = NULL;
        return 0;
    }

    /* We want to return tensor data with offsets relative to the start
     * of the file, so that the user of the API is able to access tensors
     * as it iterates over them. To do so, we need to perform a full
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

    /* Read the dimentions: all the unused dimensions are set to 1. */
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
    if (*type >= GGUF_TYPE_COUNT) return 0;
    ctx->off += 4;  // Skip tensor type.
    tensor->type = *type;

    uint64_t *offset = (uint64_t*) (ctx->data+ctx->off);
    ctx->off += 8;  // Skip tensor offset.

    tensor->offset = ctx->data_off + *offset;
    tensor->weights_data = ctx->data + tensor->offset;

    /* To accurately calculate the bytes used by this tensor on the GGUF
     * file, we need to take into account that quantization methods store
     * tensors as block of N weights. So first of all we need to understand
     * the number of padding weights (since the last block may have just
     * fewer weights stored inside, but still requires to be stored to its full
     * length). Then we can do the math to see how many blocks we need, and
     * multiply by the block size to obtain the final total size. */
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
static void gguf_print_value_callback(void *privdata, uint32_t type, union gguf_value *val, uint64_t in_array, uint64_t array_len) {
    struct gguf_print_options *po = privdata;
    if (po && po->max_array_items && in_array > po->max_array_items) {
        if (in_array-1 == po->max_array_items)
            printf("... %" PRIu64 " more items of %" PRIu64 "",
                   array_len-in_array+1, array_len);
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
            printf("%" PRIu64 "", val->uint64); break;
        case GGUF_VALUE_TYPE_INT64:
            printf("%" PRId64 "", val->int64); break;
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
gguf_ctx *gguf_create(const char *filename, int flags) {
    struct gguf_header hdr;
    memcpy(&hdr.magic,"GGUF",4);
    hdr.version = 3;
    hdr.tensor_count = 0;
    hdr.metadata_kv_count = 0;

    FILE *fp = fopen(filename, flags & GGUF_OVERWRITE ? "w" : "wx");
    if (fp == NULL) return NULL;
    if (fwrite(&hdr,1,sizeof(hdr),fp) != sizeof(hdr)) {
        fclose(fp);
        return NULL;
    }
    fclose(fp);

    return gguf_open(filename);
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
    if (gguf_remap(ctx) == 0) return 0;
    ctx->header->metadata_kv_count++;
    return 1;
}

/* Append tensor metadata (but not the actual tensor weights data) to the
 * GGUF file identified by 'ctx'. */
int gguf_append_tensor_info(gguf_ctx *ctx, const char *tensorname, uint64_t namelen, uint32_t num_dim, uint64_t *dim, uint32_t type, uint64_t offset)
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
    if (gguf_remap(ctx) == 0) return 0;
    ctx->header->tensor_count++;
    return 1;
}

/* Append tensor data enforcing the GGUF file aligment.
 * The function will take care to add the padding required to start writing
 * the tensor at an alignment multiple. */
int gguf_append_tensor_data(gguf_ctx *ctx, void *tensor, uint64_t tensor_size) {
    char padding_data[1024] = {0};
    assert(sizeof(padding_data) >= ctx->alignment);

    uint64_t padding = gguf_get_alignment_padding(ctx->alignment,ctx->size);
    if (write(ctx->fd,padding_data,padding) != (ssize_t)padding) return 0;
    if (write(ctx->fd,tensor,tensor_size) != (ssize_t)tensor_size) return 0;
    if (gguf_remap(ctx) == 0) return 0;
    return 1;
}

/* ============================ GGUF dequantization ========================= */

/* This callback is used by dequantization functions to store dequantized
 * weights in a different format than f32. By default all the dequantization
 * functions will store f32 floats just just f[j] = weight, but if
 * a store callback is passed, the function will be used. */
typedef void (*store_float_callback)(void *dst, uint64_t idx, float f);

/* Callback used to store F16 when dequantizing. */
static void gguf_store_f16_callback(void *dst, uint64_t idx, float f) {
    uint16_t *f16 = dst;
    f16[idx] = to_half(f);
}

/* Callback used to store BF16 when dequantizing. */
static void gguf_store_bf16_callback(void *dst, uint64_t idx, float f) {
    uint16_t *f16 = dst;
    f16[idx] = to_brain(f);
}

/* Q8_0 blocks dequantization to floats.
 * 'dst' is supposed to have enough space for 'count' weights. */
static void gguf_q8_0_to_float(void *weights_data, void *dst, uint64_t count, store_float_callback store_callback) {
    float *f = dst;
    struct gguf_tensor_type_features *tf =
        gguf_get_tensor_type_features(GGUF_TYPE_Q8_0);

    /* Very simple layout: |16 bit scale|32 x 8bit weights|
     * Each weight is scale * quantized_weight[0..31] */
    int8_t *block = weights_data;
    uint64_t i = 0; // i-th weight to dequantize.
    while(i < count) {
        /* For each block get the scale and convert all the
         * weights in the block. */
        float scale = from_half(*((uint16_t*)block));
        for (uint32_t j = 0; j < tf->items_per_block; j++) {
            float weight = block[j+2] * scale; // j+2 to skip the scale bytes.
            if (store_callback)
                store_callback(dst,i,weight);
            else
                f[i] = weight;
            if (++i == count) break;
        }
        block += tf->bytes_per_block; // Go to the next block.
    }
}

/* Q4_K blocks dequantization to floats.
 * 'y' is supposed to have enough space for 'count' weights. */
static void gguf_q4_k_to_float(void *weights_data, void *dst, uint64_t count, store_float_callback store_callback) {
    float *f = dst;
    uint8_t *block = weights_data;
    uint64_t i = 0; // i-th weight to dequantize.
    while(i < count) {
        /* Q4_K super-blocks have 256 total weights, split in 8 sub-block.
         * Each 8 sub-blocks have a different set of scales/mins, so
         * there are 16 total values for scales/mins, but the scales/mins
         * are also quantized (6 bits each) using two different scales:
         * scale_of_scales and scale_of_mins, that are two FP16 values
         * at the start of the super block, so:
         *
         * |FP16 s_of_scales | +
         * |FP16 s_of_mins   | +
         * |16 6 bit integers d,m pairs, one per sub-block of 32 ele | +
         * |256 x 4bit weights|
         *
         * Each quantized weight 'q' is restored as:
         *
         *      w = q * scale - min;
         */
        float scales_scale = from_half(*((uint16_t*)block));
        float mins_scale  = from_half(*((uint16_t*)(block+2)));
        block += 4;

        /* Extract the 16 x 6 bit values scales-mins pairs. The
         * encoding of those values is odd because of performance
         * reasons:
         *
         *  dddddddd dddddddd dddddddd dddddddd mmmmmmmm mmmmmmmm
         *  44000000|55111111|66222222|77333333|44000000|55111111
         *
         *  mmmmmmmm mmmmmmmm mmmmdddd mmmmdddd mmmmdddd mmmmdddd
         *  66222222|77333333|44444444|55555555|66666666|77777777
         *
         * In the above diagram you can see the 12 bytes and the
         * scales/mins 6 bits encodings. */

        /* Scale scales/mins. */
        float scales[8], mins[8];
        for (int j = 0; j < 8; j++) {
            uint8_t d,m;
            if (j < 4) {
                d = block[j] & 63;
                m = block[j+4] & 63;
            } else {
                d = (block[j+4] & 0xF) | ((block[j-4] >> 6) << 4);
                m = (block[j+4] >> 4) | ((block[j-0] >> 6) << 4);
            }
            scales[j] = d * scales_scale;
            mins[j] = m * mins_scale;
        }
        block += 12; // Seek 4-bit weights start.

        /* Finally we can extract the 256 weights.
         * We process two blocks per time, because each
         * 32 bytes have 64 weights stored like this:
         * First 32 weights of the first block are the higher 4
         * bits of each byte. Second 32 weights of the second
         * block are lower 4 bits of each byte. */
        for (uint32_t b = 0; b < 8; b += 2) {
            float scale = scales[b];
            float min = mins[b];
            /* First set: higher bits. */
            for (uint32_t j = 0; j < 32; j++) {
                uint8_t w = block[j] & 0xf;
                float weight = w * scale - min;
                if (store_callback)
                    store_callback(dst,i,weight);
                else
                    f[i] = weight;
                if (++i == count) return;
            }
            /* Second set: lower bits. */
            for (uint32_t j = 0; j < 32; j++) {
                uint8_t w = block[j] >> 4;
                float weight = w * scale - min;
                if (store_callback)
                    store_callback(dst,i,weight);
                else
                    f[i] = weight;
                if (++i == count) return;
            }
            block += 32; // Skip the two processed blocks.
        }
    }
}

/* Q6_K blocks dequantization to floats.
 * 'y' is supposed to have enough space for 'count' weights. */
static void gguf_q6_k_to_float(void *weights_data, void *dst, uint64_t count, store_float_callback store_callback) {
    float *f = dst;
    uint8_t *block = weights_data;
    uint64_t i = 0; // i-th weight to dequantize.
    while(i < count) {
        /* Q6_K super-blocks have 256 total weights, split in 16 sub-block
         * of 16 elements. There are no mins, just scales. Each sub-block
         * have a block-specific scale quantized at 8 bits via a single
         * 16-bit main scale-of-scales.
         *
         * |128 bytes of lower 4 bits of quants| +
         * |64 bytes of lower 2 bits of quants| +
         * |16 bytes of 8-bit block scales | +
         * |A single FP16 value: the scale of the scales above |
         *
         * Let's call "L" the lower 4 bits array (128 bytes)
         * and "H" the higher 2 bits array (64 bytes)
         *
         * Values are logically encoded in two 128 weights clusters
         * where the first cluster is the first 64 bytes of "L" and
         * the first 32 bytes of "H".
         *
         * Higher bits of the i-th weight from 0 to 63 are stored in the
         * lower 4 bits of L[i], while higher bits of the i-th weight
         * from 64 to 127 are stored in the higher bits of L[i-64]:
         *
         * L = |64640000|65650101|66660202|...
         *
         * So this actually is: w_low = (L[i%64] >> i/64*4) & 15
         *
         * H = |96643200|97653301|98663402|...
         *
         * Higher bits of the i-th weight are arranged like that:
         *
         * From 0 to 31,  bits 0,1 of H[i]
         * From 32 to 63, bits 3,2 of H[i-32]
         * From 64 to 95, bits 5,4 of H[i-64]
         * From 96 to 127, bits 7,6 of H[i-96]
         *
         * So this actually is: w_high = ((H[i%32] >> i/32*2) & 3) << 2
         * The same is true with the next 128 weights cluster, but
         * everything is relative to the second half of H and L.
         *
         * Finally, there is to extract the scale from the
         * 16 blocks scales array. Scales are just sequential,
         * so the i-th weight uses the scale[i/16].
         *
         * Important: In Q6_K the 6-bit quants are wisely stored
         * as unsigned integers + 32, so that there is no need to
         * do sign bit extension in order to convert the 6-bit value
         * into 8 bit value. Instead the values from -32 to 31 are
         * remapped in the 0-63 range (just adding 32).
         */
        float super_scale = from_half(*((uint16_t*)(block+128+64+16)));
        uint8_t *L = block;
        uint8_t *H = block+128;
        int8_t *scales = (int8_t*)block+128+64;
        for (int cluster = 0; cluster < 2; cluster++) {
            for (uint64_t j = 0; j < 128; j++) {
                float weight =
                      (super_scale * scales[j/16]) *
                       ((int8_t)
                        ((((L[j%64] >> (j/64*4)) & 0xF) |
                         (((H[j%32] >> (j/32*2)) & 3) << 4)))-32);
                if (store_callback)
                    store_callback(dst,i,weight);
                else
                    f[i] = weight;
                if (++i == count) return;
            }
            L += 64;
            H += 32;
            scales += 8;
        }
        block += 128+64+16+2; // Go to the next block.
    }
}

/* Q2_K blocks dequantization to floats.
 * 'y' is supposed to have enough space for 'count' weights. */
static void gguf_q2_k_to_float(void *weights_data, void *dst, uint64_t count, store_float_callback store_callback) {
    float *f = dst;
    uint8_t *block = weights_data;
    uint64_t i = 0; // i-th weight to dequantize.
    while(i < count) {
        /* Q2_K superblocks of 256 weights:
         * | 16 bytes of 16 scales, 16 mins quantized at 4 bits       | +
         * | 64 bytes of 2-bit 256 quants (16 elements x 16 blocks)  | +
         * | 2 bytes F16 scale of scales                              | +
         * | 2 bytes F16 scale of mins                                |
         *
         * Weights are organized as follows:
         *
         *                               |76543210| (bit number)
         * 16 bytes scales/mins are just |min scal| x 16, from block
         * 0 to 15, sequentially.
         *
         * 64 bytes of 2 bits quants are stored like that:
         * Weights from 0 to 31: bits 1,0 of bytes 0-31 (block 0, 1)
         * Weights from 32 to 63: bits 3,2 of bytes 0-31 (block 2, 3)
         * Weights from 64 to 95: bits 5,4 of bytes 0-31 (block 4, 5)
         * Weights from 96 to 127: bits 7,6 of bytes 0-31 (block 6, 7)
         *
         * The same happens for the next 8 blocks, stored in the remaining
         * 32 bytes.
         *
         * The final weight is computed as: w = q2 * block_scale - block_min.
         *
         * Since in this code we want to be simple more than fast (at least
         * for now), the i-th weight can be found (considering we have
         * two clusters of 128 weights each):
         *
         * cluster = i/128 # Cluster 0 or 1
         * byte = i % 32
         * shift = i / 32 * 2
         * w[i] = (quants[byte + (cluster*32)] >> shift) & 3
         */
        float scale_of_scales = from_half(*((uint16_t*)(block+16+64)));
        float scale_of_mins = from_half(*((uint16_t*)(block+16+64+2)));

        float scale = 0, min = 0;
        int bn = 0; // Block number
        for (uint64_t cluster = 0; cluster < 2; cluster++) {
            for (uint64_t j = 0; j < 128; j++) {
                /* Use new scale/min for each 16 weights sub-block. */
                if (j % 16 == 0) {
                    scale = scale_of_scales * (block[bn] & 0xf);
                    min = scale_of_mins * (block[bn] >> 4);
                    bn++;
                }
                uint8_t q = (block[16+j%32+cluster*32] >> (j/32*2)) & 3;
                float weight = q * scale - min;
                if (store_callback)
                    store_callback(dst,i,weight);
                else
                    f[i] = weight;
                if (++i == count) return;
            }
        }
        block += 16+64+4;
    }
}

/* Q4_0 blocks dequantization to floats.
 * 'dst' is supposed to have enough space for 'count' weights. */
static void gguf_q4_0_to_float(void *weights_data, void *dst, uint64_t count, store_float_callback store_callback) {
    float *f = dst;
    struct gguf_tensor_type_features *tf =
        gguf_get_tensor_type_features(GGUF_TYPE_Q4_0);

    /* Very simple layout: |16 bit scale|32 x 4bit weights|
     * Each weight is scale * (quantized_weight[0..31] - 8) */
    uint8_t *block = weights_data;
    uint64_t i = 0; // i-th weight to dequantize.
    while(i < count) {
        /* For each block get the scale and convert all the
         * weights in the block. */
        float scale = from_half(*((uint16_t*)block));
        /* First 16 weights are in the lower bits */
        for (uint32_t j = 0; j < 16; j++) {
            uint8_t value = block[j+2]; // j+2 to skip the scale bytes.
            value &= 0xf;  // lower bits
            float weight = ((int8_t) value - 8) * scale;
            if (store_callback)
                store_callback(dst,i,weight);
            else
                f[i] = weight;
            if (++i == count) break;
        }
        /* Last 16 weights are in the higher bits */
        for (uint32_t j = 0; j < 16; j++) {
            uint8_t value = block[j+2]; // j+2 to skip the scale bytes.
            value >>= 4;  // higher bits
            float weight = ((int8_t) value - 8) * scale;
            if (store_callback)
                store_callback(dst,i,weight);
            else
                f[i] = weight;
            if (++i == count) break;
        }
        block += tf->bytes_per_block; // Go to the next block.
    }
}

/* Q4_1 blocks dequantization to floats.
 * 'dst' is supposed to have enough space for 'count' weights. */
static void gguf_q4_1_to_float(void *weights_data, void *dst, uint64_t count, store_float_callback store_callback) {
    float *f = dst;
    struct gguf_tensor_type_features *tf =
        gguf_get_tensor_type_features(GGUF_TYPE_Q4_1);

    /* Very simple layout: |16 bit scale|16 bit bias|32 x 4bit weights|
     * Each weight is scale * quantized_weight[0..31] + bias */
    uint8_t *block = weights_data;
    uint64_t i = 0; // i-th weight to dequantize.
    while(i < count) {
        /* For each block get the scale and convert all the
         * weights in the block. */
        float scale = from_half(*((uint16_t*)block));
        float bias = from_half(*((uint16_t*)block+1));
        /* First 16 weights are in the lower bits */
        for (uint32_t j = 0; j < 16; j++) {
            uint8_t value = block[j+4]; // j+2 to skip the scale and bias bytes.
            value &= 0xf;  // lower bits
            float weight = value * scale + bias;
            if (store_callback)
                store_callback(dst,i,weight);
            else
                f[i] = weight;
            if (++i == count) break;
        }
        /* Last 16 weights are in the higher bits */
        for (uint32_t j = 0; j < 16; j++) {
            uint8_t value = block[j+4]; // j+2 to skip the scale and bias bytes.
            value >>= 4;  // higher bits
            float weight = value * scale + bias;
            if (store_callback)
                store_callback(dst,i,weight);
            else
                f[i] = weight;
            if (++i == count) break;
        }
        block += tf->bytes_per_block; // Go to the next block.
    }
}

/* FP16 blocks dequantization to floats.
 * 'y' is supposed to have enough space for 'count' weights. */
static void gguf_f16_to_float(void *weights_data, void *dst, uint64_t count,
                              store_float_callback store_callback) {
    float *f = dst;
    uint64_t i = 0; // i-th weight to dequantize.
    uint16_t *w16 = weights_data;
    while(i < count) {
        float weight = from_half(w16[i]);
        if (store_callback)
            store_callback(dst,i,weight);
        else
            f[i] = weight;
        i++;
    }
}

/* BF16 blocks dequantization to floats.
 * 'y' is supposed to have enough space for 'count' weights. */
static void gguf_bf16_to_float(void *weights_data, void *dst, uint64_t count,
                               store_float_callback store_callback) {
    float *f = dst;
    uint64_t i = 0; // i-th weight to dequantize.
    uint16_t *w16 = weights_data;
    while(i < count) {
        float weight = from_brain(w16[i]);
        if (store_callback)
            store_callback(dst,i,weight);
        else
            f[i] = weight;
        i++;
    }
}

/* Convert the specified tensor (quantized or not) into an array of
 * floats. The array is allocated with malloc(). If the tensor is already
 * in FP32 floats format, it is just memcpy()-ed to the destination array.
 *
 * On OOM, NULL is returned. If the tensor format is not yet supported,
 * NULL is returned as well, but errno is set to EINVAL. */
float *gguf_tensor_to_float(gguf_tensor *tensor) {
    float *f = malloc(tensor->num_weights*sizeof(float));
    if (!f) return NULL;
    if (tensor->type == GGUF_TYPE_F32) {
        memcpy(f, tensor->weights_data, tensor->num_weights*sizeof(float));
    } else if (tensor->type == GGUF_TYPE_F16) {
        gguf_f16_to_float(tensor->weights_data, f, tensor->num_weights, NULL);
    } else if (tensor->type == GGUF_TYPE_BF16) {
        gguf_bf16_to_float(tensor->weights_data, f, tensor->num_weights, NULL);
    } else if (tensor->type == GGUF_TYPE_Q8_0) {
        gguf_q8_0_to_float(tensor->weights_data, f, tensor->num_weights, NULL);
    } else if (tensor->type == GGUF_TYPE_Q4_K) {
        gguf_q4_k_to_float(tensor->weights_data, f, tensor->num_weights, NULL);
    } else if (tensor->type == GGUF_TYPE_Q6_K) {
        gguf_q6_k_to_float(tensor->weights_data, f, tensor->num_weights, NULL);
    } else if (tensor->type == GGUF_TYPE_Q2_K) {
        gguf_q2_k_to_float(tensor->weights_data, f, tensor->num_weights, NULL);
    } else if (tensor->type == GGUF_TYPE_Q4_0) {
        gguf_q4_0_to_float(tensor->weights_data, f, tensor->num_weights, NULL);
    } else if (tensor->type == GGUF_TYPE_Q4_1) {
        gguf_q4_1_to_float(tensor->weights_data, f, tensor->num_weights, NULL);
    } else {
        free(f);
        errno = EINVAL;
        return NULL;
    }
    return f;
}

/* Same as gguf_tensor_to_float() but the result will be an f16 tensor, that is
 * an array of int16_t values. */
int16_t *gguf_tensor_to_f16(gguf_tensor *tensor) {
    int16_t *f16 = malloc(tensor->num_weights*sizeof(int16_t));
    if (!f16) return NULL;
    if (tensor->type == GGUF_TYPE_F32) {
        float *f = (float*)tensor->weights_data;
        for (uint64_t j = 0; j < tensor->num_weights; j++)
            f16[j] = to_half(f[j]);
    } else if (tensor->type == GGUF_TYPE_F16) {
        memcpy(f16, tensor->weights_data, tensor->num_weights*sizeof(int16_t));
    } else if (tensor->type == GGUF_TYPE_BF16) {
        gguf_bf16_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_f16_callback);
    } else if (tensor->type == GGUF_TYPE_Q8_0) {
        gguf_q8_0_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_f16_callback);
    } else if (tensor->type == GGUF_TYPE_Q4_K) {
        gguf_q4_k_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_f16_callback);
    } else if (tensor->type == GGUF_TYPE_Q6_K) {
        gguf_q6_k_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_f16_callback);
    } else if (tensor->type == GGUF_TYPE_Q2_K) {
        gguf_q2_k_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_f16_callback);
    } else if (tensor->type == GGUF_TYPE_Q4_0) {
        gguf_q4_0_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_f16_callback);
    } else if (tensor->type == GGUF_TYPE_Q4_1) {
        gguf_q4_1_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_f16_callback);
    } else {
        free(f16);
        errno = EINVAL;
        return NULL;
    }
    return f16;
}

/* Same as gguf_tensor_to_float() but the result will be an bf16 tensor, that is
 * an array of int16_t values. */
int16_t *gguf_tensor_to_bf16(gguf_tensor *tensor) {
    int16_t *f16 = malloc(tensor->num_weights*sizeof(int16_t));
    if (!f16) return NULL;
    if (tensor->type == GGUF_TYPE_F32) {
        float *f = (float*)tensor->weights_data;
        for (uint64_t j = 0; j < tensor->num_weights; j++)
            f16[j] = to_half(f[j]);
    } else if (tensor->type == GGUF_TYPE_F16) {
        gguf_f16_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_bf16_callback);
    } else if (tensor->type == GGUF_TYPE_BF16) {
        memcpy(f16, tensor->weights_data, tensor->num_weights*sizeof(int16_t));
    } else if (tensor->type == GGUF_TYPE_Q8_0) {
        gguf_q8_0_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_bf16_callback);
    } else if (tensor->type == GGUF_TYPE_Q4_K) {
        gguf_q4_k_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_bf16_callback);
    } else if (tensor->type == GGUF_TYPE_Q6_K) {
        gguf_q6_k_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_bf16_callback);
    } else if (tensor->type == GGUF_TYPE_Q2_K) {
        gguf_q2_k_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_bf16_callback);
    } else if (tensor->type == GGUF_TYPE_Q4_0) {
        gguf_q4_0_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_bf16_callback);
    } else if (tensor->type == GGUF_TYPE_Q4_1) {
        gguf_q4_1_to_float(tensor->weights_data, f16, tensor->num_weights, gguf_store_bf16_callback);
    } else {
        free(f16);
        errno = EINVAL;
        return NULL;
    }
    return f16;
}
