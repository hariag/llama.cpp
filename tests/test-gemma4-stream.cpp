#include "llama.h"
#include "mtmd.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

int main(int argc, char ** argv) {
    if (argc != 3) { std::fprintf(stderr,"usage: %s MODEL MMPROJ\n", argv[0]); return 2; }
    llama_backend_init();
    try {
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(argv[1], mp);
        check(model, "model load failed");
        auto cp = mtmd_context_params_default(); cp.use_gpu = false; cp.n_threads = 4; cp.warmup = false;
        mtmd_context * ctx = mtmd_init_from_file(argv[2], model, cp);
        check(ctx && mtmd_support_audio_stream(ctx), "Gemma4 streaming audio unavailable");
        const int rate = mtmd_get_audio_sample_rate(ctx);
        const size_t dims = llama_model_n_embd_inp(model);
        // Include a partial frame and a 30-second segment boundary.
        for (double seconds : { 0.07, 1.31, 3.19, 30.10 }) {
            const size_t n_samples = (size_t) (seconds * rate);
            std::vector<float> samples(n_samples);
            for (size_t i=0; i<n_samples; ++i) samples[i] = 0.2f * std::sin(2.0 * 3.141592653589793 * 431.0 * i / rate);
            mtmd_bitmap * bmp = mtmd_bitmap_init_from_audio(n_samples, samples.data());
            mtmd_input_chunks * chunks = mtmd_input_chunks_init();
            const char * marker = mtmd_default_marker();
            mtmd_input_text text = { marker, std::strlen(marker), false, false };
            const mtmd_bitmap * bitmaps[] = { bmp };
            check(mtmd_tokenize(ctx, chunks, &text, bitmaps, 1) == 0, "tokenize failed");
            std::vector<float> offline;
            size_t leading_markers = 0;
            bool seen_audio = false;
            for (size_t i=0; i<mtmd_input_chunks_size(chunks); ++i) {
                const auto * chunk=mtmd_input_chunks_get(chunks,i);
                if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                    if (!seen_audio) leading_markers += mtmd_input_chunk_get_n_tokens(chunk);
                    continue;
                }
                seen_audio = true;
                check(mtmd_encode_chunk(ctx, chunk) == 0, "offline encode failed");
                size_t n=mtmd_input_chunk_get_n_tokens(chunk)*dims;
                const float * out=mtmd_get_output_embd(ctx);
                offline.insert(offline.end(), out, out+n);
            }
            check(!offline.empty(), "no offline audio rows");
            auto sp=mtmd_audio_stream_params_default();
            mtmd_audio_stream * stream=mtmd_audio_stream_init(ctx,sp);
            check(stream, "stream init failed");
            std::vector<float> streamed;
            size_t cursor=0; bool flushed=false; bool first=true; bool ended=false;
            for (size_t steps=0; steps<10000; ++steps) {
                auto before=mtmd_audio_stream_get_n_rows(stream);
                int nr=mtmd_audio_stream_step(stream,LLAMA_TOKEN_NULL);
                auto count=mtmd_audio_stream_get_n_rows(stream)-before;
                if (nr == -1) { ended=true; break; }
                check(nr >= 0 && count >= 0 && count <= nr, "stream step failed");
                if (nr) {
                    auto * out=mtmd_audio_stream_get_output_embd(stream);
                    const size_t leading = first ? leading_markers : 0;
                    check(leading + (size_t)count <= (size_t)nr, "invalid marker placement");
                    streamed.insert(streamed.end(), out + leading*dims, out+(leading+count)*dims);
                    first=false;
                } else if (cursor<n_samples) {
                    const size_t take=std::min((size_t)(rate*0.25), n_samples-cursor);
                    check(mtmd_audio_stream_push(stream,samples.data()+cursor,take)==0,"push failed"); cursor+=take;
                } else if (!flushed) {
                    check(mtmd_audio_stream_flush(stream,true)==0,"flush failed"); flushed=true;
                } else throw std::runtime_error("stream did not finish");
            }
            check(ended, "stream exceeded step limit");
            check(streamed.size()==offline.size(), "offline and stream output sizes differ");
            double mse=0, signal=0, max_abs=0;
            for(size_t i=0;i<offline.size();++i){ double diff=offline[i]-streamed[i];mse+=diff*diff;signal+=double(offline[i])*offline[i];max_abs=std::max(max_abs,std::fabs(diff)); }
            const double rel=std::sqrt(mse/std::max(1e-30,signal));
            std::printf("%.2fs: samples=%zu rows=%zu rel_rmse=%g max_abs=%g\n",seconds,n_samples,offline.size()/dims,rel,max_abs);
            check(rel<0.02, "offline and streaming embeddings differ excessively");
            mtmd_audio_stream_free(stream);mtmd_input_chunks_free(chunks);mtmd_bitmap_free(bmp);
        }
        mtmd_free(ctx);llama_model_free(model);llama_backend_free();return 0;
    } catch(const std::exception & e) { std::fprintf(stderr,"FAIL: %s\n",e.what());return 1; }
}
