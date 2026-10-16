#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"

#include <clocale>
#include <string>
#include <vector>

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    // Opt-in atomic tokenization of control strings: set
    // LLAMA_TOKENIZE_PARSE_SPECIAL=1 to make chat-template tokens like
    // <|im_start|> / <|im_end|> / <tool_call> tokenize as a single id instead
    // of being byte-split. Default behaviour (env var unset) is unchanged.
    const char * env_parse_special = std::getenv("LLAMA_TOKENIZE_PARSE_SPECIAL");
    const bool parse_special = env_parse_special != nullptr &&
                               env_parse_special[0] != '\0' && env_parse_special[0] != '0';
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, parse_special);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    base_callback_data cb_data;

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = common_debug_cb_eval<false>;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
