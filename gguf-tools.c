#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "gguflib.h"

/* ========================== Utility functions  ============================ */

/* Glob-style pattern matching. Return 1 on match, 0 otherwise. */
int strmatch(const char *pattern, int patternLen,
             const char *string, int stringLen, int nocase)
{
    while(patternLen && stringLen) {
        switch(pattern[0]) {
        case '*':
            while (patternLen && pattern[1] == '*') {
                pattern++;
                patternLen--;
            }
            if (patternLen == 1)
                return 1; /* match */
            while(stringLen) {
                if (strmatch(pattern+1, patternLen-1,
                             string, stringLen, nocase))
                    return 1; /* match */
                string++;
                stringLen--;
            }
            return 0; /* no match */
            break;
        case '?':
            string++;
            stringLen--;
            break;
        case '[':
        {
            int not, match;

            pattern++;
            patternLen--;
            not = pattern[0] == '^';
            if (not) {
                pattern++;
                patternLen--;
            }
            match = 0;
            while(1) {
                if (pattern[0] == '\\' && patternLen >= 2) {
                    pattern++;
                    patternLen--;
                    if (pattern[0] == string[0])
                        match = 1;
                } else if (pattern[0] == ']') {
                    break;
                } else if (patternLen == 0) {
                    pattern--;
                    patternLen++;
                    break;
                } else if (patternLen >= 3 && pattern[1] == '-') {
                    int start = pattern[0];
                    int end = pattern[2];
                    int c = string[0];
                    if (start > end) {
                        int t = start;
                        start = end;
                        end = t;
                    }
                    if (nocase) {
                        start = tolower(start);
                        end = tolower(end);
                        c = tolower(c);
                    }
                    pattern += 2;
                    patternLen -= 2;
                    if (c >= start && c <= end)
                        match = 1;
                } else {
                    if (!nocase) {
                        if (pattern[0] == string[0])
                            match = 1;
                    } else {
                        if (tolower((int)pattern[0]) == tolower((int)string[0]))
                            match = 1;
                    }
                }
                pattern++;
                patternLen--;
            }
            if (not)
                match = !match;
            if (!match)
                return 0; /* no match */
            string++;
            stringLen--;
            break;
        }
        case '\\':
            if (patternLen >= 2) {
                pattern++;
                patternLen--;
            }
            /* fall through */
        default:
            if (!nocase) {
                if (pattern[0] != string[0])
                    return 0; /* no match */
            } else {
                if (tolower((int)pattern[0]) != tolower((int)string[0]))
                    return 0; /* no match */
            }
            string++;
            stringLen--;
            break;
        }
        pattern++;
        patternLen--;
        if (stringLen == 0) {
            while(*pattern == '*') {
                pattern++;
                patternLen--;
            }
            break;
        }
    }
    if (patternLen == 0 && stringLen == 0)
        return 1;
    return 0;
}

/* ======================= Main and CLI options parsing ===================== */

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

    /* Show all the tensors. */
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
