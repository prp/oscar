/* The four symbols ggml-quants.c needs from ggml.c, supplied here so the
 * quantiser links without ggml.c's backend/threading tree. Sizes cover
 * exactly the dtypes oscar writes; anything else aborts loudly. Correctness
 * of these constants is pinned end-to-end by the golden-vector test against
 * the ggml reference binary. */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "ggml.h"

void ggml_abort(const char *file, int line, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ggml-quants %s:%d: ", file, line);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    abort();
}

static size_t block_size(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: return 256;
        case GGML_TYPE_Q5_0: return 32;
        default: ggml_abort(__FILE__, __LINE__, "unsupported type %d", (int)t);
    }
}

size_t ggml_type_size(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_Q5_0: return 22;
        default: ggml_abort(__FILE__, __LINE__, "unsupported type %d", (int)t);
    }
}

size_t ggml_row_size(enum ggml_type t, int64_t ne) {
    return ggml_type_size(t) * (size_t)ne / block_size(t);
}

const char *ggml_type_name(enum ggml_type t) {
    (void)t;
    return "k-quant";
}
