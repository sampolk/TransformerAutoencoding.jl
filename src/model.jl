#   src/model.jl 

# Definition of transformer-based seq2seq autoencoders, embedding and reconstruction 
# functionality, and positional encoding. 

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

using Flux
using ChainRulesCore: @non_differentiable, ignore_derivatives

# -------------------------
# Positional encoding
# -------------------------

"""
    positional_encoding(d_model, T) -> Matrix{Float32} (d_model, T)

Deterministic sinusoidal positional encoding (CPU).
- Returned shape is (d_model, T) to match (d_model, T, B) tensor conventions.
"""
function positional_encoding(d_model::Int, T::Int)
    pe = zeros(Float32, d_model, T)
    @inbounds for pos in 0:(T - 1)
        for i in 0:(div(d_model, 2) - 1)
            denom = 10000f0 ^ (2f0 * i / d_model)
            pe[2i + 1, pos + 1] = sin(pos / denom)
            pe[2i + 2, pos + 1] = cos(pos / denom)
        end
    end
    return pe
end

# Treat positional encodings as constants for autodiff
@non_differentiable positional_encoding(::Int, ::Int)

# Cache stores both CPU Matrix and GPU CuArray.
# Keyed by (d_model, T, :cpu|:cuda).
const _PE_CACHE = Dict{Tuple{Int,Int,Symbol}, Any}()

"""
    positional_encoding_cached(d_model, T, device) -> (d_model, T)

Fetch a cached positional encoding on the requested device.
- `device` is `:cpu` or `:cuda`
- Cache mutation is wrapped so autodiff never traces Dict operations.
"""
function positional_encoding_cached(d_model::Int, T::Int, device::Symbol)
    key = (d_model, T, device)

    if haskey(_PE_CACHE, key)
        return _PE_CACHE[key]
    end

    pe = positional_encoding(d_model, T)
    pe = device === :cuda ? CUDA.cu(pe) : pe

    # Avoid differentiating through cache writes 
    ignore_derivatives() do
        _PE_CACHE[key] = pe
    end

    return pe
end

# Treat caching as non-differentiable
@non_differentiable positional_encoding_cached(::Int, ::Int, ::Symbol)

# -------------------------
# Interface
# -------------------------

abstract type AbstractSeq2SeqAutoencoder end

# Helpers to normalize shapes:
# - training typically uses (D, T, B)
# - user input may be (D, T) for a single sequence
_ensure_3d(X::AbstractMatrix) = reshape(X, size(X, 1), size(X, 2), 1)
_ensure_3d(X::AbstractArray{<:Any,3}) = X
_squeeze_batch(X::AbstractArray{<:Any,3}) = size(X, 3) == 1 ? dropdims(X; dims = 3) : X

# CUDA check across CUDA.jl versions.
# Marked non-differentiable since it should not participate in gradients.
_is_cuda_array(x) = @static isdefined(CUDA, :iscuda) ? CUDA.iscuda(x) : (x isa CUDA.CuArray)
@non_differentiable _is_cuda_array(::Any)

# -------------------------
# Transformer block
# -------------------------

struct TransformerBlock
    ln1::Flux.LayerNorm
    mha::Flux.MultiHeadAttention
    drop_attn::Flux.Dropout
    ln2::Flux.LayerNorm
    ffn::Flux.Chain
    drop_ffn::Flux.Dropout
end

Flux.@layer TransformerBlock

function TransformerBlock(d_model::Int; nheads::Int = 4, ff_mult::Int = 4, dropout::Float64 = 0.0)
    @assert d_model % nheads == 0 "d_model must be divisible by nheads"

    ln1 = Flux.LayerNorm(d_model)
    mha = Flux.MultiHeadAttention(d_model; nheads = nheads, dropout_prob = dropout)
    drop_attn = Flux.Dropout(dropout)

    ln2 = Flux.LayerNorm(d_model)
    ffn = Flux.Chain(
        Flux.Dense(d_model, ff_mult * d_model, Flux.gelu),
        Flux.Dense(ff_mult * d_model, d_model),
    )
    drop_ffn = Flux.Dropout(dropout)

    return TransformerBlock(ln1, mha, drop_attn, ln2, ffn, drop_ffn)
end

function (b::TransformerBlock)(x::AbstractArray)
    # x is (d_model, T, B)
    x1 = b.ln1(x)
    y, _ = b.mha(x1)            # self-attention
    x = x .+ b.drop_attn(y)     # residual

    x2 = b.ln2(x)
    y2 = b.ffn(x2)
    x = x .+ b.drop_ffn(y2)     # residual

    return x
end

# -------------------------
# Transformer Autoencoder (attention)
# -------------------------

"""
    TransformerAutoencoder

Seq2seq autoencoder:
- Encoder: in_proj + Transformer blocks + mean pool over time into an embedding space
- Decoder: uses (Z + positional encoding) + Transformer blocks + out_proj to reconstruct sequences
"""
struct TransformerAutoencoder <: AbstractSeq2SeqAutoencoder
    in_proj::Flux.Dense
    enc::Flux.Chain
    dec::Flux.Chain
    out_proj::Flux.Dense
    d_model::Int
end

Flux.@layer TransformerAutoencoder

"""
    build_autoencoder(input_dim; d_model=64, num_layers=2, num_heads=4, dropout=0.0, ff_mult=4)

Construct a TransformerAutoencoder for sequences shaped (input_dim, T) or (input_dim, T, B).
"""
function build_autoencoder(
    input_dim::Int;
    d_model::Int = 64,
    num_layers::Int = 2,
    num_heads::Int = 4,
    dropout::Float64 = 0.0,
    ff_mult::Int = 4,
)
    in_proj = Flux.Dense(input_dim, d_model)

    enc = Flux.Chain(
        (TransformerBlock(d_model; nheads = num_heads, ff_mult = ff_mult, dropout = dropout) for _ in 1:num_layers)...,
    )
    dec = Flux.Chain(
        (TransformerBlock(d_model; nheads = num_heads, ff_mult = ff_mult, dropout = dropout) for _ in 1:num_layers)...,
    )

    out_proj = Flux.Dense(d_model, input_dim)

    return TransformerAutoencoder(in_proj, enc, dec, out_proj, d_model)
end

# -------------------------
# Autoencoding
# -------------------------
# Supports batch processing

function encode_sequence(m::TransformerAutoencoder, X::AbstractArray)
    X3 = _ensure_3d(X)                   # (input_dim, T, B)
    H0 = m.in_proj(X3)                   # (d_model, T, B)
    T = size(H0, 2)

    # Match positional encoding device to the activations.
    device = _is_cuda_array(H0) ? :cuda : :cpu
    pe = positional_encoding_cached(m.d_model, T, device)
    pe3 = reshape(pe, m.d_model, T, 1)   # broadcast over batch

    H = m.enc(H0 .+ pe3)
    z = mean(H; dims = 2)                # (d_model, 1, B)
    z = dropdims(z; dims = 2)            # (d_model, B)

    # Return a vector for the common unbatched case.
    return size(z, 2) == 1 ? vec(z) : z
end

function decode_embedding(m::TransformerAutoencoder, z::AbstractVector, T::Int)
    @assert length(z) == m.d_model
    Z = reshape(z, m.d_model, 1, 1)      # (d_model, 1, 1)

    device = _is_cuda_array(Z) ? :cuda : :cpu
    pe = positional_encoding_cached(m.d_model, T, device)
    pe3 = reshape(pe, m.d_model, T, 1)

    # Broadcast embedding over time by adding positional encoding.
    H0 = Z .+ pe3                        # (d_model, T, 1)
    H = m.dec(H0)
    Xhat = m.out_proj(H)                 # (input_dim, T, 1)
    return _squeeze_batch(Xhat)          # (input_dim, T)
end

function _decode_embeddings(m::TransformerAutoencoder, Z::AbstractMatrix, T::Int)
    # Batched decode: Z is (d_model, B)
    @assert size(Z, 1) == m.d_model
    B = size(Z, 2)
    Z3 = reshape(Z, m.d_model, 1, B)     # (d_model, 1, B)

    device = _is_cuda_array(Z3) ? :cuda : :cpu
    pe = positional_encoding_cached(m.d_model, T, device)
    pe3 = reshape(pe, m.d_model, T, 1)

    H0 = Z3 .+ pe3                       # (d_model, T, B)
    H = m.dec(H0)
    return m.out_proj(H)                 # (input_dim, T, B)
end

function reconstruct_sequence(m::AbstractSeq2SeqAutoencoder, X::AbstractArray)
    # Handles both (D, T) and (D, T, B)
    X3 = _ensure_3d(X)
    T = size(X3, 2)

    Z = encode_sequence(m, X3)
    if Z isa AbstractVector
        return decode_embedding(m, Z, T)
    else
        Xhat3 = _decode_embeddings(m, Z, T)
        return _squeeze_batch(Xhat3)
    end
end

"""
    embed_sequences(m, Xs; batchsize=64, to_cpu=true)

Embed a dataset of sequences `Xs` (each (D, Tᵢ)) into a matrix (d_model, N).
Uses padding + minibatching to avoid per-sequence overhead.
"""
function embed_sequences(
    m::AbstractSeq2SeqAutoencoder,
    Xs::Vector{<:AbstractMatrix};
    batchsize::Int = 64,
    to_cpu::Bool = true,
    rng::AbstractRNG = Random.default_rng(),
)
    Z_chunks = Matrix{Float32}[]
    for (Xb, _) in minibatches(Xs; batchsize = batchsize, shuffle = false, rng = rng)
        Z = encode_sequence(m, Xb)                 # (d_model, B)
        Zm = Z isa AbstractVector ? reshape(Z, :, 1) : Z
        push!(Z_chunks, to_cpu ? Array(Zm) : Zm)   # clustering and other later analysis is typically done on CPU
    end
    return hcat(Z_chunks...)
end
