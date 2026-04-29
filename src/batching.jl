#   src/batching.jl
#
# Variable-length batching utilities that pad (D, Tᵢ) sequences into (D, Tmax, B) 
# minibatches and return per-sequence lengths for masking.

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
    pad_sequences(Xs; pad_value=0f0, T_max=nothing) -> (Xpad, lengths)

Pad a vector of sequences `Xs` (each `(D, Tᵢ)`) into a single batch tensor `(D, Tmax, B)`.

- `lengths[b] == Tᵦ` is the true (unpadded) length of sequence `b`.
- `pad_value` fills timesteps `t > Tᵦ` and is intended to be ignored via a mask in the loss.
- If `T_max === nothing`, uses `maximum(Tᵢ)` within this batch.
"""
function pad_sequences(
    Xs::Vector{<:AbstractMatrix};
    pad_value::Float32 = 0f0,
    T_max::Union{Nothing,Int} = nothing,
)
    @assert !isempty(Xs)

    # All sequences must share the same feature dimension D.
    D = size(Xs[1], 1)

    # True lengths for masking / loss normalization downstream.
    lengths = [size(X, 2) for X in Xs]

    # Pad to max length in batch unless user overrides.
    Tmax = isnothing(T_max) ? maximum(lengths) : T_max

    B = length(Xs)

    # Allocate padded batch tensor (D, Tmax, B).
    Xpad = fill(pad_value, D, Tmax, B) # Array{Float32,3}

    @inbounds for b in 1:B
        X = Xs[b]
        @assert size(X, 1) == D "All sequences must have same feature dimension D"
        Tb = size(X, 2)

        # Fill valid timesteps; remaining timesteps stay at pad_value.
        # `@views` avoids allocating the slice.
        @views Xpad[:, 1:Tb, b] .= Float32.(X)
    end

    return (Xpad, lengths)
end

"""
    minibatches(Xs; batchsize=32, shuffle=true, rng=Random.default_rng(), pad_value=0f0)

Lazy iterator over minibatches of padded sequences.

Yields `(Xbatch, lengths)` where:
- `Xbatch` is `(D, Tmax, B)` Float32
- `lengths` are the true lengths for masking (one per sequence in the batch)
"""
function minibatches(
    Xs::Vector{<:AbstractMatrix};
    batchsize::Int = 32,
    shuffle::Bool = true,
    rng::AbstractRNG = Random.default_rng(),
    pad_value::Float32 = 0f0,
)
    N = length(Xs)

    idx = collect(1:N)
    if shuffle
        Random.shuffle!(rng, idx)
    end

    return (
        pad_sequences(Xs[idx[i:min(i + batchsize - 1, N)]]; pad_value = pad_value) for
        i in 1:batchsize:N
    )
end
