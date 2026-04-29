#   examples/toy_gridworld.jl
#
# End-to-end example demonstrating TransformerAutoencoding.jl on a small synthetic,
# variable-length sequence dataset.

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
using Optimisers
using Dates
using CUDA

# Toy dataset utilities (guarded to avoid redefinition warnings in the same REPL session)
if !isdefined(@__MODULE__, :GridworldUtils)
    include("gridworld_utils.jl")
end
using .GridworldUtils: build_gridworld_dataset

println("1. Loaded packages.")

# ----------------------------
# hyperparameters
# ----------------------------
# GPU toggle is advisory; training will only use GPU if CUDA is installed and functional.
use_gpu = true

# Set to true if you would like TensorBoard logging during training
use_tblogger = true

# Training loop settings.
num_epochs    = 200
learning_rate = 1e-3
log_interval  = 10  # write logs every N epochs (if tb_logdir is set)

# Minibatching/padding settings
batchsize = 64
pad_value = 0f0

# Transformer autoencoder architecture parameters
d_model    = 32
num_layers = 2
num_heads  = 4
dropout    = 0.1

# ----------------------------
# logging/output paths
# ----------------------------
repo_root = normpath(joinpath(@__DIR__, ".."))
run_stamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")

# Base directory for this example's outputs
run_root = joinpath(repo_root, "runs", "toy_gridworld", run_stamp)
isdir(run_root) || mkpath(run_root)

# Per-run TB directory (or `nothing` to disable)
tb_logdir = (use_tblogger && log_interval > 0) ? joinpath(run_root, "tb") : nothing

# ----------------------------
# data
# ----------------------------
# Generate labeled state-action trajectories for a small gridworld exploration
seqs, labels = build_gridworld_dataset(; cluster_size = 10, seed = 0)
println("2. Created toy data.")

# Convert to Vector{Matrix{Float32}} where each sequence is (D, T), and normalize features.
Xs, μ, σ = prepare_sequence_dataset(seqs; normalize = true)
input_dim = size(Xs[1], 1)

# Inspect variable sequence lengths.
Ts = size.(Xs, 2)
println("3. Prepared dataset: N=$(length(Xs)), input_dim=$input_dim, variable T in [$(minimum(Ts)), $(maximum(Ts))]")

# ----------------------------
# train
# ----------------------------
# Build a transformer seq2seq autoencoder for inputs of dimension D.
ae = build_autoencoder(
    input_dim;
    d_model    = d_model,
    num_layers = num_layers,
    num_heads  = num_heads,
    dropout    = dropout,
)

# Train using the single training function. `lossfn` takes (Xhat3, X3, lengths).
fit = train_autoencoder!(
    ae,
    Xs;
    epochs       = num_epochs,
    opt_rule     = Optimisers.Adam(learning_rate),
    log_interval = log_interval,
    use_gpu      = use_gpu,
    tb_logdir    = tb_logdir,   # set to `nothing` to disable logging
    return_cpu   = true,        # keep the returned model CPU-safe for serialization and clustering
    batchsize    = batchsize,
    pad_value    = pad_value,
    lossfn       = masked_mse_loss,  # masked loss ignores padding timesteps
)

ae_fit = fit.model

println("4. Trained on device = ", fit.device)
if !isempty(fit.losses)
    println("   First logged loss = ", first(fit.losses))
    println("   Last  logged loss = ", last(fit.losses))
end

# Reconstruction statistics computed on the original sequence data.
μmse, mn, mx = reconstruction_mse_stats(ae_fit, Xs)
println("   Mean reconstruction MSE over dataset = ", μmse)
println("   Min  reconstruction MSE = ", mn)
println("   Max  reconstruction MSE = ", mx)

# ----------------------------
# clustering
# ----------------------------
# Embed sequences into a fixed-size representation Z of shape (d_model, N).
Z = embed_sequences(ae_fit, Xs; batchsize = batchsize)

# Choose K by silhouette score and return the best kmeans result + score.
best_res, best_K, best_score = best_kmeans_clustering(Z; Kmax = 6)
println("\nBest K = $best_K, silhouette = $best_score")

# Compare discovered clusters to the ground-truth midpoint label for interpretability.
comp = cluster_composition(best_res.assignments, labels; K = best_K)

println("\nCluster composition by true midpoint label:")
for k in 1:best_K
    println("  Cluster $k: count=$(count(==(k), best_res.assignments)), midpoint counts=$(comp[k])")
end

println("\nOutputs written under: $run_root")