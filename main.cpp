#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>

#include "src/loader/parameters.h"
#include "src/model/mistral/modules.h"

// RNG + sampling helpers
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> distr(0.0, 1.0);

uint32_t sample_max(InferenceState& infer) {
    float max_val = infer.logits.data[0];
    size_t res = 0;
    for (size_t i = 0; i < infer.config.vocab_size; i++) {
        if (infer.logits.data[i] > max_val) {
            res = i;
            max_val = infer.logits.data[i];
        }
    }
    return static_cast<uint32_t>(res);
}

uint32_t sample_multinomial(InferenceState& infer, float temp) {
    if (temp > 0.0f) {
        // scale logits by temperature
        for (int i = 0; i < static_cast<int>(infer.logits.numel); i++) {
            infer.logits.data[i] /= temp;
        }
        softmax(infer.probs, infer.logits);
    } else {
        // pure greedy
        return sample_max(infer);
    }

    float r = static_cast<float>(distr(gen));
    float total = 0.0f;

    for (int i = 0; i < static_cast<int>(infer.probs.numel); i++) {
        total += infer.probs.data[i];
        if (total >= r) {
            return static_cast<uint32_t>(i);
        }
    }
    return static_cast<uint32_t>(infer.probs.numel - 1);
}

template <typename T>
uint32_t generate(Model<T>& model, InferenceState& infer, size_t token) {
    model.forward(infer, token);
    // temp = 0 => greedy
    return sample_multinomial(infer, 0.0f);
}

template <typename T>
void run_generation(Model<T>& model,
                    InferenceState& infer,
                    Parameters& params,
                    const std::string& prompt) {
    // Encode prompt
    std::vector<uint32_t> got = params.tokenizer.encode(prompt);

    if (got.empty()) {
        std::cout << "Prompt encoded to 0 tokens, nothing to do.\n";
        return;
    }

    // Debug
#if 1
    model.forward(infer, got[0]);

    float max_logit = infer.logits.max(); // uses Tensor<float>::max()
    std::cout << "Logits max after first token: " << max_logit << "\n";

    std::cout << "First 10 logits: ";
    int to_show = std::min<int>(10, static_cast<int>(infer.logits.numel));
    for (int i = 0; i < to_show; i++) {
        std::cout << infer.logits.data[i] << " ";
    }
    std::cout << "\n";
#endif

    // Feed remaining prompt tokens except the last one
    for (size_t i = 1; i + 1 < got.size(); i++) {
        model.forward(infer, got[i]);
    }

    // Start generation from last prompt token
    uint32_t t = got.back();

    for (int i = 0; i < 50; i++) {
        t = generate<T>(model, infer, t);

        // decode single token and stream it out
        std::cout << params.tokenizer.decode({ t }) << std::flush;
    }

    std::cout << std::endl;
}

int main(int argc, char** argv) {
#if 1
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <prompt>\n";
        return 1;
    }
#else
    // Useful for running in visual studio
    char* test_argv[] = {
        argv[0],  // keep program name
        (char*)"D:/my_files/my_docs/ai/models/torchless/mistral-int8.bin",
        (char*)"paris is the capital of"
    };
    argv = test_argv;
    argc = 3;
#endif

    std::string model_path = argv[1];
    std::string prompt     = argv[2];

    // Load parameters
    auto params = std::make_shared<Parameters>();
    params->load_parameters(model_path);

    // DEBUG: print quant mode
    std::cout << "Quant mode from header: " << params->config.quant << "\n";

    // Create inference state
    InferenceState infer(params->config);

    if (params->config.quant == "int8") {
        std::cout << "Using INT8 model (Model<int8_t>)\n";
        Model<int8_t> model(params);
        run_generation<int8_t>(model, infer, *params, prompt);
    } else {
        std::cout << "Using F32 model (Model<float>)\n";
        Model<float> model(params);
        run_generation<float>(model, infer, *params, prompt);
    }

    return 0;
}
