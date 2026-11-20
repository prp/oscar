#!/usr/bin/env python3
"""Emit OSCAR's fork diff as a patch series, one patch per SUBJECT.

The submodules point at proper upstream and this series is applied after
checkout. The split is by subject rather than by fork or by commit for one
reason — when the pin moves, a conflict should land in one body of work.

    scripts/make_patch_series.py            # write oscar/patches/
    scripts/make_patch_series.py --verify   # also prove the series rebuilds HEAD
"""
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "patches")

# (label, path, base ref). Both forks branch from a pinned upstream ref, which
# is what makes the series exist at all.
FORKS = [
    ("mistral.rs", "third_party/oscar-mistral.rs",
     "962112d18ea5729fe7acaae5085ce54c2fb66dab"),
    ("candle", "third_party/oscar-candle", "0.10.2"),
]

# Subject -> exact paths, in reading order. Exact, not globs: the script asserts
# that every changed path lands in exactly one subject, so a file added
# upstream-side by a future bump fails loudly here instead of being silently
# dropped from the series. Path order is the order hunks appear in the patch, so
# list the contract first and its call sites after.
SUBJECTS = {
    "mistral.rs": [
        ("pluggable-decode-and-quant-kernels",
         "The APIs the fork opens so oscar can drive it: the engine's step "
         "taken through oscar's own round, where prefill may break, and the "
         "quant-side kernels oscar's models reach.", [
             "mistralrs/src/lib.rs",
             "mistralrs/src/gguf.rs",
             "mistralrs-quant/src/lib.rs",
             "mistralrs-quant/src/gguf/mod.rs",
             "mistralrs-quant/src/unquantized/mod.rs",
             "mistralrs-quant/src/metal_kernels/mod.rs",
         ]),
        ("external-gguf-architectures",
         "The registry an out-of-crate architecture joins through, and reading "
         "the metadata and tensors it needs — including the shards OSCAR pages "
         "itself rather than letting the loader map.", [
             "mistralrs-core/src/gguf/mod.rs",
             "mistralrs-core/src/gguf/content.rs",
             "mistralrs-core/src/gguf/gguf_tokenizer.rs",
             "mistralrs-core/src/utils/gguf_metadata.rs",
             "mistralrs-core/src/utils/model_config.rs",
             "mistralrs-core/src/utils/mod.rs",
             "mistralrs-core/src/pipeline/gguf.rs",
             "mistralrs-core/src/pipeline/paths.rs",
         ]),
        ("kv-offload-and-recurrent-state",
         "The KV spill seam, and the recurrent-state pool the hybrid models "
         "carry beside their KV.", [
             "mistralrs-core/src/kv_cache/mod.rs",
             "mistralrs-core/src/kv_cache/hybrid_cache.rs",
             # The per-layer recurrent override the hybrid cache config
             # grew; these four in-tree models construct the config, so
             # they carry its new (empty) field.
             "mistralrs-core/src/models/granite.rs",
             "mistralrs-core/src/models/qwen3_next.rs",
             "mistralrs-core/src/vision_models/qwen3_5/text.rs",
             "mistralrs-core/src/vision_models/qwen3_5_moe/text.rs",
         ]),
        ("cancellation-admission-and-paged-scheduling",
         "Admission, cancel-on-disconnect, and the scheduling changes a "
         "paging engine needs. Woven into the engine loop — no seam exists "
         "for this, so it is carried or upstreamed.", [
             "mistralrs-core/src/scheduler/mod.rs",
             "mistralrs-core/src/scheduler/default_scheduler.rs",
             "mistralrs-core/src/paged_attention/config.rs",
             "mistralrs-core/src/paged_attention/mod.rs",
             "mistralrs-core/src/paged_attention/kv_memory.rs",
             "mistralrs-core/src/paged_attention/cache_engine.rs",
             "mistralrs-core/src/paged_attention/block_pool.rs",
             "mistralrs-core/src/paged_attention/kv_cache_manager.rs",
             "mistralrs-core/src/paged_attention/scheduler.rs",
             "mistralrs-core/src/sequence.rs",
             "mistralrs-core/src/engine/add_request.rs",
             "mistralrs-core/src/engine/mod.rs",
             "mistralrs-core/src/pipeline/speculative.rs",
             # A request allocates no contiguous KV up front, so every
             # pipeline's cache reset loses the flag that loaded it and
             # gemma4 the per-layer predicate that sized it.
             "mistralrs-core/src/pipeline/amoe.rs",
             "mistralrs-core/src/pipeline/diffusion.rs",
             "mistralrs-core/src/pipeline/embedding.rs",
             "mistralrs-core/src/pipeline/ggml.rs",
             "mistralrs-core/src/pipeline/multimodal.rs",
             "mistralrs-core/src/pipeline/normal.rs",
             "mistralrs-core/src/pipeline/speech.rs",
             "mistralrs-core/src/vision_models/gemma4/text.rs",
         ]),
        ("tool-calls-while-reasoning",
         "Tool-call parsing while the model is still reasoning, the "
         "sampling-site hooks the eval recorder needs, and an ordered tool "
         "schema — the template renders it into the prompt, so a map whose "
         "iteration order varies per instance re-renders the tools block on "
         "every request and defeats prefix reuse.", [
             "mistralrs-core/src/tools/mod.rs",
             "mistralrs-mcp/src/tools.rs",
             "mistralrs-mcp/src/client.rs",
             "mistralrs-core/src/search/mod.rs",
             "mistralrs-core/src/pipeline/sampling.rs",
         ]),
        ("vendor-sampling-defaults",
         "A sampling field the client omitted is answered by the MODEL's own "
         "shipped `generation_config.json` before the API's default can stand "
         "in for it, applied once on the single path every HTTP handler "
         "already sends through. Without it an OpenAI client that sends no "
         "temperature is served at 1.0 with every truncation disabled, which "
         "it cannot see and did not ask for.", [
             "mistralrs-core/src/sampler.rs",
             "mistralrs-server-core/src/handler_core.rs",
         ]),
        ("reasoning-channels-assembled-in-template",
         "A reasoning format whose channel delimiters the template assembles "
         "from markers instead of writing literally, so neither the tag "
         "detector nor the parser can be driven off the delimiters themselves "
         "(Kimi K3).", [
             "mistralrs-core/src/reasoning_parsers/tag_based.rs",
             "mistralrs-core/src/reasoning_parsers/mod.rs",
             "mistralrs-core/src/pipeline/chat_template.rs",
         ]),
        ("prefix-cache-reuse-policy",
         "What the prefix cache keeps, when it declines to, the seam an "
         "out-of-crate store spills it to, and how much of a prompt it spared "
         "the model — reported to the caller where OpenAI puts it.", [
             "mistralrs-core/src/prefix_cacher.rs",
             "mistralrs-core/src/response.rs",
             "mistralrs/src/agent.rs",
         ]),
        ("attention-and-layer-primitives",
         "Shared layer primitives and the attention entry points the ports "
         "call.", [
             "mistralrs-core/src/layers.rs",
         ]),
        ("text-only-vision-packaging",
         "Loading a Qwen3.5 MoE checkpoint whose quantized packaging ships no "
         "vision encoder: the encoder becomes optional and image or video input "
         "fails with a reason instead of a missing tensor.", [
             "mistralrs-core/src/vision_models/qwen3_5_moe/mod.rs",
         ]),
        ("module-visibility-and-dependencies",
         "Where the architectures are named to the rest of the crate, and the "
         "dependency edges that make oscar reachable from the fork.", [
             "Cargo.toml",
             "mistralrs-cli/Cargo.toml",
             "mistralrs-core/Cargo.toml",
             "mistralrs-core/build.rs",
             "mistralrs-core/src/lib.rs",
             "mistralrs-core/src/pipeline/mod.rs",
             "mistralrs/src/model.rs",
         ]),
        ("paged-rows-as-kernel-inputs",
         "The engine's scheduled rows as the paged kernels' inputs: one row per "
         "query token, a block table and a visible-key count per row, the slot "
         "mappings uploaded once per device.", [
             "mistralrs-core/src/pipeline/inputs_processor.rs",
         ]),
        ("serve-request-path",
         "What a request carries into the chat template, and the "
         "one-line-in one-line-out record the serve lane keeps of it. Two "
         "things a template needs and the message types did not hand it: tool "
         "call arguments as a map rather than a JSON string, and the "
         "assistant's own reasoning, which every vendor template renders back "
         "into history and no model could ever see, because neither the wire "
         "type nor the in-process message list declared a field for it. Plus "
         "the ids the server advertises for the model it loaded, each naming "
         "the generation mode it serves, so a client pins a reference to what "
         "is loaded rather than a copy of it. And the record says when a "
         "request produced NO ANSWER: a reply whose content channel is empty "
         "because the whole budget went to reasoning is indistinguishable "
         "from a refusal otherwise, and on the streamed path only a field "
         "carried across the chunks can tell them apart. And the system "
         "message a chat that sends none is given: the embedder's default, "
         "installed once, so the model is told what it is. And the phase the "
         "engine is in and the tokens it has sampled, read by the status line "
         "beside a server. And a request for a model id the server does not "
         "serve is refused before it is parsed, with the status and the error "
         "object OpenAI gives it and one line saying which ids are served.", [
             "mistralrs-core/src/request.rs",
             "mistralrs-core/src/live.rs",
             "mistralrs-server-core/src/util.rs",
             "mistralrs-server-core/src/openai.rs",
             "mistralrs-server-core/src/chat_completion.rs",
             "mistralrs-server-core/src/completions.rs",
             "mistralrs-server-core/src/streaming.rs",
             "mistralrs-server-core/src/handlers.rs",
             "mistralrs-server-core/src/responses.rs",
             "mistralrs-server-core/src/cached_responses.rs",
             "mistralrs/src/messages.rs",
         ]),
        ("a-request-seeds-its-own-sampling",
         "An OpenAI `seed` on a request seeds a sampler RNG of the sequence's "
         "own, so the same request draws the same tokens and a served reply "
         "can be banked byte for byte; unseeded, a sequence draws from the "
         "engine's shared RNG as before. `SamplingParams` carries the field, "
         "so every literal of it names it — here, the crates oscar does not "
         "build, so the workspace stays whole.", [
             "mistralrs-bench/src/main.rs",
             "mistralrs-cli/src/commands/bench.rs",
             "mistralrs-cli/src/commands/run/interactive.rs",
             "mistralrs-pyo3/src/lib.rs",
             "mistralrs-server/src/interactive_mode.rs",
         ]),
    ],
    "candle": [
        ("metal-stream-and-buffer-pool-seam",
         "The stream machinery OSCAR drives its own encoder scopes through, "
         "the command-buffer observer, and THE SEAM EVERY METAL BUFFER "
         "COMES THROUGH: candle asks an installed pool for every allocation "
         "and has no path to the driver beside it, naming which pool a "
         "request belongs to, the queue whose free lists serve it, the "
         "fence a buffer the CPU writes into is reused behind, the fence a "
         "freed buffer's PAGES go back to the OS on (which an idle queue can "
         "only ever meet the second way), the LINE OF CODE a host fill came "
         "from — the tensor constructors are `#[track_caller]` down to the "
         "seam, so the pool holds the caller's own location as the buffer's "
         "label where there is no kernel to name it — and the one "
         "call a model build places a weight it KEEPS through — the floor, "
         "from host bytes, refusing a device source.", [
             "candle-metal-kernels/src/metal/command_buffer.rs",
             "candle-core/src/metal_backend/device.rs",
             "candle-core/src/metal_backend/thread_stream.rs",
             "candle-metal-kernels/src/kernels/sort.rs",
             "candle-metal-kernels/src/metal/pool.rs",
             "candle-metal-kernels/src/metal/commands.rs",
             "candle-metal-kernels/src/metal/encoder.rs",
             "candle-metal-kernels/src/metal/device.rs",
             "candle-metal-kernels/src/metal/buffer.rs",
             "candle-metal-kernels/src/metal/compute_pipeline.rs",
             "candle-metal-kernels/src/kernel.rs",
             "candle-metal-kernels/src/metal/mod.rs",
             "candle-core/src/metal_backend/mod.rs",
             "candle-core/src/backend.rs",
             "candle-core/src/device.rs",
             "candle-core/src/tensor.rs",
             "candle-core/src/dummy_metal_backend.rs",
             "candle-metal-kernels/Cargo.toml",
         ]),
        ("quantized-dtypes-and-buffers",
         "The three i-quant dtypes a GGUF reader must name and size but has "
         "no codec for, the Metal buffer accessors oscar reaches quantized "
         "weights through, a weight quantised on the host placed once "
         "with no device storage made first, and the GGUF reader's staging "
         "as an anonymous mapping rather than a malloc block the allocator "
         "would keep dirty.", [
             "candle-core/src/quantized/metal.rs",
             "candle-core/src/quantized/mod.rs",
             "candle-core/src/quantized/dummy_metal.rs",
             "candle-core/src/quantized/gguf_file.rs",
         ]),
        ("paged-moe-matmul-kernels",
         "The pooled and with-hids MoE variants, threaded inside upstream's "
         "own kernel_mul_mm / kernel_mul_mm_id bodies, and the tensor-op GEMM "
         "in MLX's NAX structure over the same dequantisers (dense and both "
         "routed entries, half activations the host casts once per input). "
         "38e's target, and a rewrite rather than a move — nothing relocates "
         "these. Also the q8_0/q4_K matvec geometry, whose dispatch table "
         "disagreed with the kernel bodies it dispatches.", [
             "candle-metal-kernels/src/kernels/quantized.rs",
             "candle-metal-kernels/src/metal_src/quantized.metal",
         ]),
        ("strided-copy-and-unary-kernels",
         "The templated 3D-grid rewrite of the strided copy path, and the "
         "unary kernels beside it.", [
             "candle-metal-kernels/src/kernels/cast.rs",
             "candle-metal-kernels/src/kernels/unary.rs",
             "candle-metal-kernels/src/metal_src/cast.metal",
             "candle-metal-kernels/src/metal_src/unary.metal",
         ]),
        ("nn-helpers",
         "The one single-site change that belongs to no larger body.", [
             "candle-nn/src/ops.rs",
         ]),
        ("nax-tensor-op-kernels",
         "MLX's NAX fragment and tile helpers (mlx b6368984, nax.metal, "
         "prepended to the sdpa and quantized sources) and its NAX attention "
         "kernels beside the steel one, dispatched as MLX does — head dims 128 "
         "and 256 on a bf16 query, the head-dim split kernel at 256; the "
         "tensor-op GEMM over the fork's dequantisers rides the quantized "
         "subject. Also the steel kernel's causal K/V loop bound, clamped to "
         "the key tile count: unclamped, a tile-unaligned query length ran "
         "the loop past kL and read beyond the K/V buffers, invisible until "
         "the bytes past a buffer were NaN.", [
             "candle-metal-kernels/src/source.rs",
             "candle-metal-kernels/src/metal_src/nax.metal",
             "candle-metal-kernels/src/kernels/sdpa.rs",
             "candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal",
         ]),
        ("deterministic-random-seed",
         "A random draw is a function of its seed alone: the seed advances in a "
         "dispatch of its own behind the draw (stored from inside it, the next "
         "seed was read by the threads scheduled after the store), and a thread "
         "past the half writes nothing (the grid is rounded up to whole "
         "threadgroups, and those threads wrote the middle a second time).", [
             "candle-metal-kernels/src/metal_src/random.metal",
             "candle-metal-kernels/src/kernels/random.rs",
         ]),
    ],
}


#: Build products the checkout regenerates; never fork source.
GENERATED = {"Cargo.lock"}


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", repo, *args],
                          capture_output=True, text=True, check=check).stdout


# The diff's format spelled out, so the series is the same bytes on every
# machine whatever its git configuration says about prefixes, the algorithm,
# context or abbreviation: a `--check` compares bytes.
DIFF = ["diff", "--no-renames", "--src-prefix=c/", "--dst-prefix=w/",
        "--diff-algorithm=histogram", "--indent-heuristic", "-U3", "--abbrev=8"]


def build(fork, repo, base, verbose, outdir=None):
    # INTENT-TO-ADD the files Oscar creates. The submodule is pristine
    # upstream, so those files are untracked and `git diff` cannot see them at
    # all — without this the series silently loses every new file it should
    # carry. Reset at the end of the run so the index is left as found.
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").split()
    if untracked:
        git(repo, "add", "-N", *untracked)
    changed = set(git(repo, *DIFF, "--name-only", base).split())
    # Cargo rewrites the fork's lockfile whenever it resolves the workspace
    # through oscar's path overrides, so it is a build product of the checkout
    # rather than fork source. Carrying it would put a machine's dependency
    # resolution in the series; refusing on it blocks the gate suite for a file
    # no subject wants.
    changed -= GENERATED
    subjects = SUBJECTS[fork]

    # Every changed path in exactly one subject, and no subject naming a path
    # that is not in the diff. Both directions, because a silent gap here is a
    # patch series that does not rebuild the tree.
    assigned = [p for _, _, paths in subjects for p in paths]
    dupes = {p for p in assigned if assigned.count(p) > 1}
    missing = changed - set(assigned)
    stale = set(assigned) - changed
    for label, items in (("in two subjects", dupes), ("in no subject", missing),
                         ("named but unchanged", stale)):
        if items:
            print(f"{fork}: {len(items)} paths {label}:", file=sys.stderr)
            for p in sorted(items):
                print(f"    {p}", file=sys.stderr)
            sys.exit(1)

    outdir = os.path.join(outdir or OUT, fork)
    os.makedirs(outdir, exist_ok=True)
    for stale_patch in os.listdir(outdir):
        if stale_patch.endswith(".patch"):
            os.remove(os.path.join(outdir, stale_patch))

    written = []
    for i, (name, blurb, paths) in enumerate(subjects, start=1):
        # Per path, in the order the subject lists them: a reader should meet the
        # contract before its call sites, which git's own path ordering will not do.
        diff = "".join(git(repo, *DIFF, base, "--", p) for p in paths)
        header = [
            f"Subject: [{fork}] {name}",
            "",
            *[f"  {line}" for line in blurb.split(". ") if line],
            "",
            f"Applies to: {base}",
            f"Files: {len(paths)}",
            "",
            "---",
            "",
        ]
        path = os.path.join(outdir, f"{i:04d}-{name}.patch")
        with open(path, "w") as fh:
            fh.write("\n".join(header))
            fh.write(diff)
        written.append(path)
        if verbose:
            print(f"  {i:04d}-{name}.patch  {len(paths):3d} files")
    git(repo, "reset", "-q")
    return written


def verify(fork, repo, base):
    """Apply the series to a clean checkout of the base and compare trees.

    Compares only what the series claims: GENERATED paths are excluded from
    it, so comparing them would fail the moment anything is built."""
    with tempfile.TemporaryDirectory() as tmp:
        wt = os.path.join(tmp, "wt")
        git(repo, "worktree", "add", "--detach", "-f", wt, base)
        try:
            outdir = os.path.join(OUT, fork)
            for patch in sorted(os.listdir(outdir)):
                if not patch.endswith(".patch"):
                    continue
                r = subprocess.run(["git", "-C", wt, "apply", "--3way",
                                    os.path.join(outdir, patch)],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"{fork}: {patch} FAILED to apply\n{r.stderr}", file=sys.stderr)
                    return False
            git(wt, "add", "-A")
            got = git(wt, "write-tree").strip()
            git(repo, "add", "-A")
            want = git(repo, "write-tree").strip()
            git(repo, "reset", "-q")
            differ = {line.split("\t", 1)[1]
                      for line in git(repo, "diff-tree", "-r", "--name-status",
                                      got, want).splitlines() if line}
            ok = not (differ - GENERATED)
            print(f"{fork}: series rebuilds the patched worktree — {'YES' if ok else f'NO ({got} != {want})'}")
            return ok
        finally:
            subprocess.run(["git", "-C", repo, "worktree", "remove", "--force", wt],
                           capture_output=True)


def applies_to(series_dir):
    """The ref the series in `series_dir` was cut against, from its first patch's header."""
    for name in sorted(os.listdir(series_dir)):
        if not name.endswith(".patch"):
            continue
        with open(os.path.join(series_dir, name)) as fh:
            for line in fh:
                if line.startswith("Applies to: "):
                    return line[len("Applies to: "):].strip()
                if line.startswith("diff --git"):
                    break
    return None


def differing_files(a, b):
    """The paths whose sections differ between the patch files `a` and `b`; "the header" for a difference
    before the first diff."""
    def sections(path):
        out, key = {}, "the header"
        for line in open(path, encoding="utf-8", errors="replace"):
            if line.startswith("diff --git "):
                key = line.split()[2][2:]
            out[key] = out.get(key, "") + line
        return out
    sa, sb = sections(a), sections(b)
    return sorted(k for k in sa.keys() | sb.keys() if sa.get(k) != sb.get(k))


def check(only=None, against=None):
    """Is the committed series what the fork trees would produce right now?

    The trees are the working copy and `patches/` is the committed form, so they
    can silently disagree the moment a fork file is edited. Nothing else notices:
    the trees are untracked by oscar and pristine at HEAD in their own repos.

    `only` restricts the check to one fork. `against` compares with that copy of
    the fork's series instead of `patches/` — setup.sh keeps the series it
    applied beside its stamp and asks whether the tree still holds exactly it —
    and the diff is taken against the ref that copy was cut for.
    """
    forks = [f for f in FORKS if only in (None, f[0])]
    if not forks:
        sys.exit(f"no fork named {only}; the forks are {', '.join(f[0] for f in FORKS)}")
    with tempfile.TemporaryDirectory() as tmp:
        for fork, rel, base in forks:
            if against:
                base = applies_to(against) or base
            build(fork, os.path.join(ROOT, rel), base, False, outdir=tmp)
        drift = []
        for fork, _, _ in forks:
            fresh_dir, canon_dir = os.path.join(tmp, fork), against or os.path.join(OUT, fork)
            fresh = set(os.listdir(fresh_dir))
            canon = set(os.listdir(canon_dir)) if os.path.isdir(canon_dir) else set()
            names = {n for n in fresh | canon if n.endswith(".patch")}
            for name in sorted(names):
                if name not in canon:
                    drift.append(f"{fork}/{name}: not in patches/ — a new subject")
                elif name not in fresh:
                    drift.append(f"{fork}/{name}: no longer produced — a stale patch")
                else:
                    files = differing_files(os.path.join(fresh_dir, name), os.path.join(canon_dir, name))
                    if files:
                        drift.append(f"{fork}/{name}: differs in {', '.join(files)}")
    if drift:
        print("the fork trees and patches/ differ:", file=sys.stderr)
        for d in drift:
            print(f"    {d}", file=sys.stderr)
        return False
    print("patches/ matches the fork trees")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="apply the series to a clean pin checkout and compare trees")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if patches/ is not what the trees would produce")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--fork", metavar="NAME",
                    help="--check one fork only: mistral.rs or candle")
    ap.add_argument("--against", metavar="DIR",
                    help="--check against this copy of the fork's series instead of patches/")
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check(args.fork, args.against) else 1)
    if args.fork or args.against:
        ap.error("--fork and --against go with --check")

    ok = True
    for fork, rel, base in FORKS:
        repo = os.path.join(ROOT, rel)
        print(f"{fork}  ({base})")
        written = build(fork, repo, base, not args.quiet)
        print(f"  {len(written)} patches")
        if args.verify:
            ok &= verify(fork, repo, base)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
