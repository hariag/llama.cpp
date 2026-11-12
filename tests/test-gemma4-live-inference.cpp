#include "llama.h"
#include "mtmd.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "usage: " << argv[0] << " MODEL MMPROJ PCM_F32_16KHZ [EXPECTED_SUBSTRING]\n";
        return 2;
    }

    try {
        std::ifstream file(argv[3], std::ios::binary | std::ios::ate);
        check(bool(file), "cannot open audio");
        const auto size = file.tellg();
        check(size > 0 && size % sizeof(float) == 0, "invalid f32 audio");
        file.seekg(0);
        std::vector<float> samples(size_t(size) / sizeof(float));
        file.read(reinterpret_cast<char *>(samples.data()), size);
        check(bool(file), "cannot read audio");

        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(argv[1], model_params);
        check(model, "model load failed");

        auto context_params = llama_context_default_params();
        context_params.n_ctx = 2048;
        context_params.n_batch = 512;
        context_params.n_threads = 4;
        context_params.n_threads_batch = 4;
        llama_context * context = llama_init_from_model(model, context_params);
        check(context, "context init failed");

        auto mtmd_params = mtmd_context_params_default();
        mtmd_params.use_gpu = false;
        mtmd_params.n_threads = 4;
        mtmd_params.warmup = false;
        mtmd_context * mtmd = mtmd_init_from_file(argv[2], model, mtmd_params);
        check(mtmd && mtmd_support_audio_stream(mtmd) && mtmd_get_audio_sample_rate(mtmd) == 16000,
              "16 kHz streaming audio unavailable");
        mtmd_audio_stream * stream = mtmd_audio_stream_init(mtmd, mtmd_audio_stream_params_default());
        check(stream, "stream init failed");

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int dimensions = llama_model_n_embd_inp(model);
        llama_pos position = 0;
        auto decode = [&](const llama_token * tokens, const float * embeddings, int count, bool logits) {
            for (int offset = 0; offset < count;) {
                const int n = std::min(512, count - offset);
                llama_batch batch = llama_batch_init(n, embeddings ? dimensions : 0, 1);
                for (int i = 0; i < n; i++) {
                    if (embeddings) {
                        std::memcpy(batch.embd + size_t(i) * dimensions,
                                    embeddings + size_t(offset + i) * dimensions, size_t(dimensions) * sizeof(float));
                    } else {
                        batch.token[i] = tokens[offset + i];
                    }
                    batch.pos[i] = position++;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = logits && offset + i == count - 1;
                }
                batch.n_tokens = n;
                const int result = llama_decode(context, batch);
                llama_batch_free(batch);
                check(result == 0, "decode failed");
                offset += n;
            }
        };
        auto decode_text = [&](const std::string & text, bool bos, bool logits) {
            const int n = -llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, bos, true);
            check(n > 0, "token count failed");
            std::vector<llama_token> tokens(n);
            check(llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), n, bos, true) == n,
                  "tokenize failed");
            decode(tokens.data(), nullptr, n, logits);
        };

        decode_text("<|turn>user\nTranscribe exactly what is spoken in this audio.\n", true, false);
        int rows_before_flush = 0;
        for (size_t offset = 0; offset < samples.size();) {
            const size_t n = std::min<size_t>(4000, samples.size() - offset); // 250 ms
            check(mtmd_audio_stream_push(stream, samples.data() + offset, n) == 0, "push failed");
            offset += n;
            while (true) {
                const int rows = mtmd_audio_stream_step(stream, LLAMA_TOKEN_NULL);
                check(rows >= 0, "stream step failed");
                if (!rows) {
                    break;
                }
                decode(nullptr, mtmd_audio_stream_get_output_embd(stream), rows, false);
                rows_before_flush += rows;
            }
        }
        check(rows_before_flush > 0, "stream emitted no rows before flush");
        check(mtmd_audio_stream_flush(stream, true) == 0, "flush failed");
        while (true) {
            const int rows = mtmd_audio_stream_step(stream, LLAMA_TOKEN_NULL);
            check(rows != -2, "flush step failed");
            if (rows <= 0) {
                break;
            }
            decode(nullptr, mtmd_audio_stream_get_output_embd(stream), rows, false);
        }

        decode_text("<turn|>\n<|turn>model\n", false, true);
        llama_sampler * sampler = llama_sampler_init_greedy();
        std::string output;
        bool stopped = false;
        for (int i = 0; i < 64; i++) {
            const llama_token token = llama_sampler_sample(sampler, context, -1);
            if (llama_vocab_is_eog(vocab, token)) {
                stopped = true;
                break;
            }
            char piece[512];
            const int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
            check(n >= 0, "token conversion failed");
            output.append(piece, n);
            decode(&token, nullptr, 1, true);
        }
        std::cout << "rows_before_flush=" << rows_before_flush << " output=" << output << std::endl;
        check(stopped, "generation did not finish");
        check(argc < 5 || output.find(argv[4]) != std::string::npos, "expected transcription not found");

        llama_sampler_free(sampler);
        mtmd_audio_stream_free(stream);
        mtmd_free(mtmd);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
