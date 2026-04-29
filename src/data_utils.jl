#   ./src/data_utils.jl
# Dataset preparation utilities for converting common sequence formats into 
# Vector{Matrix{Float32}} and optional feature normalization.

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

"""
    prepare_sequence_dataset(seqs; normalize=false) -> (Xs, μ, σ)

Convert a collection of sequences into a vector of `(D, T)` Float32 matrices to be used in autoencoding. 

Supported inputs:
1) `seqs::Vector{<:AbstractMatrix}`:
   Each element is already a `(D, T)` matrix (any numeric element type).

2) `seqs::Vector{<:Vector{<:AbstractVector}}`:
   Each element is a sequence (length T) of feature vectors (length D), e.g.
   `seq[t] == [f1, f2, ..., fD]`.

If `normalize=true`, normalization is computed feature-wise across all timesteps in
the entire dataset (across all sequences), then applied to each sequence.

Returns:
- `Xs::Vector{Matrix{Float32}}`: sequences as Float32 matrices
- `μ::Vector{Float32}`: per-feature mean used for normalization (zeros if normalize=false)
- `σ::Vector{Float32}`: per-feature std used for normalization (ones if normalize=false)
"""
function prepare_sequence_dataset(seqs; normalize::Bool = false)
    # Convert input into a consistent internal representation: Vector{Matrix{Float32}}
    Xs = _to_matrices(seqs)

    # Feature dimension (rows). We assume sequences are non-empty.
    D = size(Xs[1], 1)

    # Default μ/σ if not normalizing
    μ = zeros(Float32, D)
    σ = ones(Float32, D)

    if !normalize
        return (Xs, μ, σ)
    end

    # Stack all sequences along time to compute global per-feature statistics.
    # allX has shape (D, sum(Tᵢ)).
    allX = hcat(Xs...)
    μ = vec(mean(allX; dims = 2))
    σ = vec(std(allX; dims = 2))

    # Avoid divide-by-zero for constant features.
    σ .= ifelse.(σ .== 0f0, 1f0, σ)

    # Normalize each sequence, creating a new vector rather than mutating inputs.
    Xs_norm = [(X .- μ) ./ σ for X in Xs]
    return (Xs_norm, μ, σ)
end

# ---- internals ----

# Case 1: sequences already provided as (D, T) matrices.
# We defensively convert to Matrix{Float32} to ensure downstream code is type-stable.
function _to_matrices(seqs::Vector{<:AbstractMatrix})
    return [Matrix{Float32}(X) for X in seqs]
end

# Case 2: sequences are vectors of feature vectors.
# We pack each sequence into a single (D, T) matrix.
function _to_matrices(seqs::Vector{<:Vector{<:AbstractVector}})
    first_seq = first(seqs)
    first_step = first(first_seq)
    D = length(first_step)

    Xs = Matrix{Float32}[]
    for seq in seqs
        T = length(seq)
        X = Matrix{Float32}(undef, D, T)

        # Fill one timestep per column.
        @inbounds for t in 1:T
            @assert length(seq[t]) == D "Inconsistent feature dimension at t=$t (got $(length(seq[t])) vs D=$D)"
            X[:, t] = Float32.(seq[t])
        end

        push!(Xs, X)
    end
    return Xs
end
