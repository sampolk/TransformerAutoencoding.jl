#   ./src/TransformerAutoencoding.jl

# Package entry point that defines the TransformerAutoencoding module.

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

module TransformerAutoencoding

# Load core dependencies
using Flux
using LinearAlgebra
using Random
using Statistics
using Logging
using Optimisers

# Optional runtime support:
# - CUDA is used when `use_gpu=true` in training and CUDA is available and/or functional
# - TensorBoardLogger is used when `tb_logdir` is provided during training
using CUDA
using TensorBoardLogger

# Clustering utilities used for evaluation
using Clustering
using Distances
using Statistics
using Random

# PrecompileTools is used to define a small workload to reduce first-run latency
using PrecompileTools

# Load utilities
include("data_utils.jl") # data conversion and normalization
include("batching.jl") # batching and padding utilities 
include("model.jl") # model definitions and autoencoding helpers
include("training.jl") # training loop and masked loss definintions
include("clustering.jl") #  clustering utilities for embedding analysis in examples
include("precompile.jl") #  precompile workload definitions

# ----------------------------
# Public exports
# ----------------------------

# Dataset preparation
export prepare_sequence_dataset

# Batching utilities 
export pad_sequences,
    minibatches

# Model type, constructors, and core operations
export AbstractSeq2SeqAutoencoder,
    TransformerAutoencoder,
    build_autoencoder,
    encode_sequence,
    reconstruct_sequence,
    decode_embedding,
    embed_sequences

# Training and evaluation helpers
export train_autoencoder!,
    reconstruction_mse_stats,
    reconstruction_loss_stats

# Masked losses (usable as `lossfn=...`) and mask helper
export masked_mse_loss,
    masked_bce_loss,
    masked_logitbce_loss,
    mask_from_lengths

# Clustering utilities (used to test embedding quality in examples)
export best_kmeans_clustering,
    silhouette_score,
    cluster_composition,
    clustering_accuracy

end # module
