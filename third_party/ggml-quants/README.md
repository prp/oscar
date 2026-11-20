# ggml-quants (vendored)

Verbatim files from ggml (via llama.cpp, upstream commit 56dc28a7c):
`ggml-quants.c` and its header closure. Only the k-quant reference
quantisers (`quantize_q3_K` / `q4_K` / `q5_K` / `q6_K`, with optional
imatrix weights) are linked; `shim.c` supplies the four trivial symbols
(`ggml_abort`, `ggml_type_size`, `ggml_row_size`, `ggml_type_name`)
that would otherwise pull in the whole of `ggml.c` and its backend
tree. The shim is ours; every other file is unmodified upstream, under
ggml's MIT license (`LICENSE`).

These are the same functions `llama-quantize` runs (via
`ggml_quantize_chunk`), which is what makes the `download` command's
re-quantisation produce the bytes the rest of the world's tooling
would produce. candle's non-imatrix quantiser is measurably worse
(`make_qkx1_quants` vs ggml's `make_qkx2_quants`) and is deliberately
not used for writing artifacts.
