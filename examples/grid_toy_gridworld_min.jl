#   examples/grid_toy_gridworld_min.jl

# Run a small hyperparameter sweep for TransformerAutoencoding.jl on a synthetic,
# variable-length gridworld trajectory dataset, and write per-run artifacts plus a
# single sweep summary under `runs/grid_min/`.

# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

# This material is based upon work supported by the Under Secretary of War for Research 
# and Engineering under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any 
# opinions, findings, conclusions or recommendations expressed in this material are those 
# of the author(s) and do not necessarily reflect the views of the Under Secretary of War 
# for Research and Engineering.

# © 2026 Massachusetts Institute of Technology.

# The software/firmware is provided to you on an As-Is basis

# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 
# 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government 
# rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed 
# above. Use of this work other than as specifically authorized by the U.S. Government may 
# violate any copyrights that exist in this work.

# SPDX-License-Identifier: MIT

# Written by Sam L. Polk, MIT Lincoln Laboratory

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# Ensure all dependencies in Project/Manifest are installed.
Pkg.instantiate()

using TransformerAutoencoding
using Random
using Statistics
using Optimisers
using Printf
using Dates
using Serialization
using CUDA

# Local toy-data generator
include("gridworld_utils.jl")
using .GridworldUtils: build_gridworld_dataset

# Keep the train/test split stable across runs
DATA_SPLIT_SEED = 1

# ----------------------------
# dataset
# ----------------------------
# Generate labeled time series trajectories
seqs, labels = build_gridworld_dataset(; cluster_size = 10, seed = 0)

# Convert to Vector{Matrix{Float32}} of shape (D, T) and normalize features.
Xs, μ, σ = prepare_sequence_dataset(seqs; normalize = true)
N = length(Xs)

# train/test split independent of per-run RNG
Random.seed!(DATA_SPLIT_SEED)
idx = collect(1:N)
shuffle!(idx)
ntrain = max(1, round(Int, 0.7 * N))
train_idx = idx[1:ntrain]
test_idx = idx[(ntrain + 1):end]

Xs_train = Xs[train_idx]
Xs_test  = Xs[test_idx]
labels_all = labels

println("Dataset: N=$N, train=$(length(Xs_train)), test=$(length(Xs_test))")

# ----------------------------
# grid
# ----------------------------
# Hyperparameter grid (cartesian product) for a small sweep.
d_models    = [16, 32]
num_layerss = [1, 2]
lrs         = [1e-3]
seeds       = [1, 2]

# run name tags for output directories/files
lr_tag(lr::Real) = replace(lowercase(@sprintf("%.0e", Float64(lr))), "+" => "")
make_run_name(d_model::Int, num_layers::Int, lr::Real, seed::Int) =
    "d$(d_model)_L$(num_layers)_lr$(lr_tag(lr))_seed$(seed)"

grid = [
    (; d_model, num_layers, lr, seed, run_name = make_run_name(d_model, num_layers, lr, seed))
    for d_model in d_models
    for num_layers in num_layerss
    for lr in lrs
    for seed in seeds
]

# ----------------------------
# train settings
# ----------------------------
# Enable GPU only if CUDA is installed and functional; otherwise fall back to CPU
use_gpu = CUDA.has_cuda() && CUDA.functional()

# Set to true if you would like tensorboard logging during training
use_tblogger = true

epochs       = 200
log_interval = 10  # TensorBoard writes every `log_interval` epochs (if tb_logdir is set)

batchsize = 64     # padded minibatch size
pad_value = 0f0    # pad value for variable-length batching

sweep_stamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
repo_root = normpath(joinpath(@__DIR__, ".."))
outdir    = joinpath(repo_root, "runs", "grid_min")
isdir(outdir) || mkpath(outdir)

# Collect per-run summaries for a single “sweep summary” file at the end.
summary = NamedTuple[]

for (i, cfg) in enumerate(grid)
    println("\n==============================")
    println("Run $(i)/$(length(grid))  $(cfg.run_name)")
    println("==============================")

    # Per-run RNG controls model init + minibatch shuffle (keeps runs reproducible).
    rng = MersenneTwister(cfg.seed)

    # Only set TB directory if enabled; keep it unique per run invocation
    stamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS.sss")

    # One folder per hyperparameter configuration:
    #   runs/grid_min/<cfg.run_name>/
    run_dir = joinpath(outdir, cfg.run_name)
    isdir(run_dir) || mkpath(run_dir)

    # TensorBoard log directory for this run
    # TODO: we are always logging here, maybe we should allow this to be an option
    tb_logdir = (use_tblogger && log_interval > 0) ? joinpath(run_dir, "tb", stamp) : nothing

    # Build a transformer seq2seq autoencoder for inputs of dimension D.
    ae = build_autoencoder(
        size(Xs_train[1], 1);
        d_model    = cfg.d_model,
        num_layers = cfg.num_layers,
        num_heads  = 4,
        dropout    = 0.1,
    )

    # Train. `lossfn` is a callable with signature (Xhat3, X3, lengths) -> scalar.
    fit = train_autoencoder!(
        ae,
        Xs_train;
        epochs       = epochs,
        opt_rule     = Optimisers.Adam(cfg.lr),
        log_interval = log_interval,
        use_gpu      = use_gpu,
        tb_logdir    = tb_logdir,        # set to `nothing` to disable TensorBoard logging
        return_cpu   = true,             # keep saved artifacts CPU-safe
        rng          = rng,
        batchsize    = batchsize,
        pad_value    = pad_value,
        lossfn       = masked_mse_loss,
    )

    model = fit.model

    # Reconstruction quality on original (unpadded) sequences.
    train_mse = reconstruction_mse_stats(model, Xs_train)
    test_mse  = reconstruction_mse_stats(model, Xs_test)

    # Embed all sequences for clustering (returns (d_model, N) by default).
    Z = embed_sequences(model, Xs; batchsize = batchsize)

    # Choose K using silhouette score, then evaluate clustering vs ground-truth labels.
    best_res, best_K, best_sil = best_kmeans_clustering(Z; Kmax = 6, rng = rng)
    acc, mapping = clustering_accuracy(best_res.assignments, labels_all)

    # Serialize a compact run artifact (avoid storing TB logger objects).
    run_outfile = joinpath(run_dir, "run_" * stamp * ".jls")

    run_results = (;
        cfg,
        model,
        fit = (; losses = fit.losses, device = fit.device),
        train_mse,
        test_mse,
        best_K,
        best_sil,
        acc,
        mapping,
        labels_all,
    )
    Serialization.serialize(run_outfile, run_results)

    println("Saved: $run_outfile")
    println("  device = $(fit.device)")
    println("  train mse (mean/min/max) = $(train_mse)")
    println("  test  mse (mean/min/max) = $(test_mse)")
    println("  best_K = $best_K, silhouette = $best_sil")
    println("  clustering acc = $acc")
    println("  cluster->label mapping = $mapping")

    # Per-run summary intended for quick inspection
    push!(summary, (;
        cfg,
        outfile = run_outfile,
        device  = fit.device,
        train_mse,
        test_mse,
        best_K,
        best_sil,
        acc,
        mapping,
    ))

    # Help GC between runs
    local ae, fit, Z, best_res, best_K
end

# Save one file with all run summaries
summary_file = joinpath(outdir, "grid_summary.jls")
Serialization.serialize(summary_file, summary)
println("\nWrote summary: $summary_file")