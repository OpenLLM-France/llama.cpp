#include "debug.h"

#include "log.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <string>

static std::string common_ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static float common_ggml_get_float_value(const uint8_t * data,
                           ggml_type       type,
                           const size_t *  nb,
                           size_t          i0,
                           size_t          i1,
                           size_t          i2,
                           size_t          i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    float  v;
    if (type == GGML_TYPE_F16) {
        v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
    } else if (type == GGML_TYPE_F32) {
        v = *(const float *) &data[i];
    } else if (type == GGML_TYPE_I64) {
        v = (float) *(const int64_t *) &data[i];
    } else if (type == GGML_TYPE_I32) {
        v = (float) *(const int32_t *) &data[i];
    } else if (type == GGML_TYPE_I16) {
        v = (float) *(const int16_t *) &data[i];
    } else if (type == GGML_TYPE_I8) {
        v = (float) *(const int8_t *) &data[i];
    } else if (type == GGML_TYPE_BF16) {
        v = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
    } else {
        GGML_ABORT("fatal error");
    }
    return v;
}

#define INDENT "    "

template <bool abort>
void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum += v;
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG(INDENT "[\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) {
                LOG(INDENT INDENT "..., \n");
                i2 = ne[2] - n;
            }
            LOG(INDENT INDENT "[\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) {
                    LOG(INDENT INDENT INDENT "..., \n");
                    i1 = ne[1] - n;
                }
                LOG(INDENT INDENT INDENT "[");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) {
                        LOG("   ..., ");
                        i0 = ne[0] - n;
                    }
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    LOG("%12.4f", v);
                    if (i0 < ne[0] - 1) {
                        LOG(", ");
                    }
                }
                LOG("  ],\n");
            }
            LOG(INDENT INDENT "],\n");
        }
        LOG(INDENT "]\n");
        LOG(INDENT "sum = %f\n", sum);
    }

    if constexpr (abort) {
        if (std::isnan(sum)) {
            LOG("encountered NaN - aborting\n");
            exit(0);
        }
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
template <bool abort_on_nan> bool common_debug_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (base_callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true;  // Always retrieve data
    }

    bool matches_filter = cb_data->tensor_filters.empty();

    if (!matches_filter) {
        for (const auto & filter : cb_data->tensor_filters) {
            if (std::regex_search(t->name, filter)) {
                matches_filter = true;
                break;
            }
        }
    }

    char src1_str[128] = { 0 };
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, common_ggml_ne_string(src1).c_str());
    }

    if (matches_filter) {
        LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__, t->name, ggml_type_name(t->type),
            ggml_op_desc(t), src0->name, common_ggml_ne_string(src0).c_str(), src1 ? src1_str : "",
            common_ggml_ne_string(t).c_str());
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type) && matches_filter) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        common_debug_print_tensor<abort_on_nan>(data, t->type, t->ne, t->nb, 3);

        // Optional full-tensor binary dump for layer-by-layer comparison work.
        // Activated by setting env var LLAMA_DUMP_TENSORS_FILE=/path/to/out.bin.
        // Optionally narrow what gets dumped with LLAMA_DUMP_TENSORS_REGEX
        // (a single regex; anchored implicitly with regex_search). If unset,
        // every tensor that already matched cb_data's filter gets dumped.
        // Per-tensor binary record (little-endian):
        //     u32 name_len, char name[name_len],
        //     u32 dtype (ggml_type), i64 ne[4],
        //     u64 n_bytes, u8 data[n_bytes]
        const char * dump_path = std::getenv("LLAMA_DUMP_TENSORS_FILE");
        if (dump_path) {
            static std::regex   dump_regex;
            static bool         dump_regex_set = false;
            static bool         dump_regex_valid = false;
            if (!dump_regex_set) {
                dump_regex_set = true;
                const char * pat = std::getenv("LLAMA_DUMP_TENSORS_REGEX");
                if (pat && *pat) {
                    try { dump_regex = std::regex(pat); dump_regex_valid = true; }
                    catch (const std::regex_error &) { dump_regex_valid = false; }
                }
            }
            bool should_dump = !dump_regex_valid || std::regex_search(t->name, dump_regex);
            if (should_dump) {
                static FILE * dump_fout = nullptr;
                static std::string opened_path;
                if (!dump_fout || opened_path != dump_path) {
                    if (dump_fout) fclose(dump_fout);
                    dump_fout = std::fopen(dump_path, "wb");
                    opened_path = dump_path;
                }
                if (dump_fout) {
                    uint32_t name_len = (uint32_t) std::strlen(t->name);
                    std::fwrite(&name_len, 4, 1, dump_fout);
                    std::fwrite(t->name, 1, name_len, dump_fout);
                    uint32_t dtype = (uint32_t) t->type;
                    std::fwrite(&dtype, 4, 1, dump_fout);
                    int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
                    std::fwrite(ne, 8, 4, dump_fout);
                    uint64_t nbytes = (uint64_t) ggml_nbytes(t);
                    std::fwrite(&nbytes, 8, 1, dump_fout);
                    std::fwrite(data, 1, nbytes, dump_fout);
                    std::fflush(dump_fout);
                }
            }
        }
    }

    return true;
}

// Explicit template instantiations
template bool common_debug_cb_eval<false>(ggml_tensor *, bool, void *);
template bool common_debug_cb_eval<true>(ggml_tensor *, bool, void *);
template void common_debug_print_tensor<false>(uint8_t *, ggml_type, const int64_t *, const size_t *, int64_t);
template void common_debug_print_tensor<true>(uint8_t *, ggml_type, const int64_t *, const size_t *, int64_t);
