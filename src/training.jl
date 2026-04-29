# src/training.jl
#
# Single training entry point: train_autoencoder!
#
# Key behavior:
# - padding and batching of sequences
# - computes masked losses so padded timesteps do not contribute to loss calculations
# - supports training with CPU and, optionally, GPU 
# - supports user-supplied masked loss functions:
#       lossfn(Xhat3, X3, lengths) -> scalar
#   where Xhat3/X3 are (D, T, B) and lengths is length B
#
# Assumes model.jl defines:
# - AbstractSeq2SeqAutoencoder
# - reconstruct_sequence(m, X) where X can be 2D or 3D

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
using Statistics
using Random
using Logging: with_logger

using CUDA
using TensorBoardLogger
using Optimisers
import Zygote

# -------------------------
# Small helpers 
# -------------------------

# Convert a scalar (or 0-d array) to Float32 on CPU for logging/statistics.
_scalar32(x::Number) = Float32(x)
_scalar32(x::AbstractArray) = Float32(only(Array(x)))  # handles 0-d arrays too

# Utilities for BCE-like targets.
_as_float32_targets(y) = eltype(y) <: AbstractFloat ? y : Float32.(y)
_prepare_bce_targets(y) = clamp.(_as_float32_targets(y), 0f0, 1f0)

# Leaf-wise moves used by Flux.fmap 
_to_cpu_leaf(x) = x
_to_cpu_leaf(x::AbstractArray) = Array(x)

_to_gpu_leaf(x) = x
_to_gpu_leaf(x::AbstractArray) = CUDA.cu(x)

# Move a model to GPU (if requested and functional), returning device tag.
function _maybe_to_gpu_model(m, use_gpu::Bool)
    if use_gpu && CUDA.has_cuda() && CUDA.functional()
        return (Flux.fmap(_to_gpu_leaf, m), :cuda)
    end
    return (m, :cpu)
end

# Move a batch tensor based on a device tag.
_maybe_to_gpu_batch(X, device::Symbol) = (device === :cuda ? CUDA.cu(X) : X)

# Optionally return the trained model on CPU for later use
_maybe_to_cpu_model(m, return_cpu::Bool) = (return_cpu ? Flux.fmap(_to_cpu_leaf, m) : m)

# Ensure 3D view for losses: (D, T, B). The trainer always feeds padded minibatches as 3D.
_as3d(A::AbstractMatrix) = reshape(A, size(A, 1), size(A, 2), 1)
_as3d(A::AbstractArray{<:Any,3}) = A

# CUDA check across CUDA.jl versions. Marked no-grad to keep autodiff out of control flow.
_cuda_iscuda(x) = @static isdefined(CUDA, :iscuda) ? CUDA.iscuda(x) : (x isa CUDA.CuArray)
Zygote.@nograd _cuda_iscuda

# -------------------------
# Masking 
# -------------------------

"""
    mask_from_lengths(lengths, T, device) -> (T, B) Float32 array on CPU/GPU

Construct a timestep mask for variable-length batches.

- `lengths[b]` is the true (unpadded) length for batch element `b`
- `mask[t,b] = 1` if `t <= lengths[b]` else `0`

This returns a Float32 mask on the requested device (`:cpu` or `:cuda`).
"""
function mask_from_lengths(lengths::AbstractVector{<:Integer}, T::Int, device::Symbol)
    B = length(lengths)
    t = reshape(1:T, T, 1)        # (T, 1)
    L = reshape(lengths, 1, B)    # (1, B)
    mask_cpu = Float32.(t .<= L)  # (T, B) on CPU
    return device === :cuda ? CUDA.cu(mask_cpu) : mask_cpu
end

# Mask depends only on data lengths, and we do not differentiate through it.
Zygote.@nograd mask_from_lengths

# -------------------------
# Built-in masked losses (exported)
# -------------------------

"""
    masked_mse_loss(Xhat3, X3, lengths)

Masked mean squared error over valid (non-padding) elements.

Inputs:
- `Xhat3`, `X3`: arrays of shape (D, T, B)
- `lengths`: vector of length B

Returns a scalar where padding contributes zero.
"""
function masked_mse_loss(Xhat3, X3, lengths)
    D, T, B = size(X3)
    device = _cuda_iscuda(Xhat3) ? :cuda : :cpu

    mask_tb = mask_from_lengths(lengths, T, device) # (T, B)
    mask3 = reshape(mask_tb, 1, T, B)               # (1, T, B)

    num = sum(((Xhat3 .- X3) .^ 2) .* mask3)
    denom = max(Float32(D * sum(lengths)), 1f0)     # normalize by number of valid scalars
    return num / denom
end

"""
    masked_bce_loss(probs3, targets3, lengths)

Masked binary cross-entropy over valid elements.

Inputs:
- `probs3`: probabilities in (0,1), shape (D, T, B)
- `targets3`: targets, shape (D, T, B)
- `lengths`: vector length B
"""
function masked_bce_loss(probs3, targets3, lengths)
    D, T, B = size(targets3)
    device = _cuda_iscuda(probs3) ? :cuda : :cpu

    mask_tb = mask_from_lengths(lengths, T, device)
    mask3 = reshape(mask_tb, 1, T, B)

    ϵ = 1f-7
    p = clamp.(probs3, ϵ, 1f0 - ϵ)
    y = clamp.(targets3, 0f0, 1f0)

    elem = -(y .* log.(p) .+ (1f0 .- y) .* log.(1f0 .- p))
    num = sum(elem .* mask3)

    denom = max(Float32(D * sum(lengths)), 1f0)
    return num / denom
end

"""
    masked_logitbce_loss(logits3, targets3, lengths)

Masked logit binary cross-entropy over valid elements.

Inputs:
- `logits3`: raw logits, shape (D, T, B)
- `targets3`: targets, shape (D, T, B)
- `lengths`: vector length B
"""
function masked_logitbce_loss(logits3, targets3, lengths)
    D, T, B = size(targets3)
    device = _cuda_iscuda(logits3) ? :cuda : :cpu

    mask_tb = mask_from_lengths(lengths, T, device)
    mask3 = reshape(mask_tb, 1, T, B)

    x = logits3
    y = clamp.(targets3, 0f0, 1f0)

    # Stable logit-BCE: max(x,0) - x*y + log(1+exp(-abs(x)))
    elem = max.(x, 0f0) .- x .* y .+ log1p.(exp.(-abs.(x)))
    num = sum(elem .* mask3)

    denom = max(Float32(D * sum(lengths)), 1f0)
    return num / denom
end

# ----------------------------
# Resolve Symbol loss -> (lossfn, pred_tf, target_tf)
# ----------------------------

# Map a Symbol to:
# - a masked loss function
# - a prediction transform applied to the model output
# - a target transform applied to the batch inputs
function _resolve_loss(loss::Symbol)
    if loss === :mse
        return (masked_mse_loss, identity, identity)
    elseif loss === :bce
        # Uses probabilities; apply sigmoid to model output.
        return (masked_bce_loss, Flux.sigmoid, _prepare_bce_targets)
    elseif loss === :logitbce || loss === :bce_logits
        # Uses logits directly; targets are clamped to [0,1].
        return (masked_logitbce_loss, identity, _prepare_bce_targets)
    else
        error("Unknown loss Symbol: $loss. Use :mse, :bce, :logitbce, or pass lossfn=... directly.")
    end
end

# ----------------------------
# Stats helpers
# ----------------------------

"""
    reconstruction_mse_stats(m, Xs) -> (mean, min, max)

Compute unmasked MSE statistics over a dataset (no padding involved).
This is a quick sanity check, not the exact training loss (training uses masks).
"""
function reconstruction_mse_stats(m::AbstractSeq2SeqAutoencoder, Xs::Vector{<:AbstractMatrix})
    mses = map(Xs) do X
        _scalar32(Flux.Losses.mse(reconstruct_sequence(m, X), X))
    end
    return (mean(mses), minimum(mses), maximum(mses))
end

"""
    reconstruction_loss_stats(m, Xs; loss=:mse, lossfn=nothing, pred_tf=identity, target_tf=identity)
        -> (mean, min, max)

Compute masked loss statistics over a dataset by treating each sequence as a batch of size 1.

- If `lossfn` is provided, it is used directly (expects 3D tensors + lengths).
- Otherwise, `loss` is resolved into a masked loss + default transforms.
"""
function reconstruction_loss_stats(
    m::AbstractSeq2SeqAutoencoder,
    Xs::Vector{<:AbstractMatrix};
    loss::Symbol = :mse,
    lossfn::Union{Nothing,Function} = nothing,
    pred_tf::Function = identity,
    target_tf::Function = identity,
)
    if isnothing(lossfn)
        lossfn_resolved, default_pred_tf, default_target_tf = _resolve_loss(loss)
        lossfn = lossfn_resolved
        pred_tf = pred_tf === identity ? default_pred_tf : pred_tf
        target_tf = target_tf === identity ? default_target_tf : target_tf
    end

    vals = map(Xs) do X
        # Treat each sequence as (D,T,B=1).
        Xhat = pred_tf(reconstruct_sequence(m, X))
        Xt = target_tf(X)

        Xhat3 = _as3d(Xhat)
        Xt3 = _as3d(Xt)
        lengths = [size(X, 2)]  # full length, B=1

        _scalar32(lossfn(Xhat3, Xt3, lengths))
    end

    return (mean(vals), minimum(vals), maximum(vals))
end

# ----------------------------
# Training function
# ----------------------------
"""
    train_autoencoder!(m, Xs; kwargs...) -> NamedTuple

Train a seq2seq autoencoder on a dataset of variable-length sequences.

Inputs
- m: model implementing `reconstruct_sequence(m, X)` for X shaped (D,T,B).
- Xs: Vector of matrices, each shaped (D, Tᵢ).

Keyword arguments
Training loop
- epochs::Int=20: number of epochs.
- opt_rule=Optimisers.Adam(1e-3): Optimisers.jl rule passed to `Flux.setup`.
- λ_reg::Float32=0f0: L2 regularization coefficient applied to `Flux.trainables(m)`.

Batching
- batchsize::Int=64: number of sequences per minibatch.
- pad_value::Float32=0f0: padding value used by `pad_sequences` for minibatches.
- shuffle_each_epoch::Bool=true: shuffle order each epoch.
- rng::AbstractRNG=Random.default_rng(): RNG used for shuffling.

Loss configuration (choose one)
- lossfn::Union{Nothing,Function}=nothing: custom loss with signature
  `lossfn(Xhat3, X3, lengths) -> scalar`, where Xhat3 and X3 are (D,T,B).
- loss::Symbol=:mse: convenience loss selector when `lossfn` is not provided;
  supported values: :mse, :bce, :logitbce.
- pred_tf::Function=identity: applied to model reconstruction before loss
  (e.g., `Flux.sigmoid` for :bce).
- target_tf::Function=identity: applied to targets before loss
  (e.g., casting/clamping for BCE variants).

Device / output placement
- use_gpu::Bool=false: if true and CUDA is available, moves model and batches to GPU.
- return_cpu::Bool=true: if true, converts the returned model back to CPU arrays.

Logging
- log_interval::Int=0: if > 0, record `losses` every `log_interval` epochs.
- tb_logdir::Union{Nothing,String}=nothing: if set and `log_interval>0`, write scalar logs.
- logger::Union{Nothing,Function}=nothing: optional callback `(epoch, loss) -> nothing`;
  if provided, it takes precedence over TensorBoard logging.

Returns
NamedTuple with fields:
- model: trained model (on CPU if return_cpu=true).
- losses: logged average epoch losses (one per log interval).
- device: :cpu or :cuda (effective device used during training).
- Xs: the input dataset reference (as provided).
- lossname: Symbol describing loss mode (:mse/:bce/:logitbce/:custom).
"""
function train_autoencoder!(
    m::AbstractSeq2SeqAutoencoder,
    Xs::Vector{<:AbstractMatrix};
    epochs::Int = 20,
    opt_rule = Optimisers.Adam(1e-3),

    λ_reg::Float32 = 0f0,

    log_interval::Int = 0,
    tb_logdir::Union{Nothing,String} = nothing,
    logger::Union{Nothing,Function} = nothing,

    use_gpu::Bool = false,
    return_cpu::Bool = true,

    rng::AbstractRNG = Random.default_rng(),
    shuffle_each_epoch::Bool = true,

    batchsize::Int = 64,
    pad_value::Float32 = 0f0,

    # Choose one of the below
    loss::Symbol = :mse,
    lossfn::Union{Nothing,Function} = nothing,

    # Optional transforms
    pred_tf::Function = identity,
    target_tf::Function = identity,
)
    # Preserve the user intent in the returned fit object.
    lossname = isnothing(lossfn) ? loss : :custom

    # Move model to desired device, if requested and available.
    m2, device = _maybe_to_gpu_model(m, use_gpu)
    Flux.trainmode!(m2)

    # If a Symbol loss is used, select the corresponding masked loss and default transforms.
    if isnothing(lossfn)
        lossfn_resolved, default_pred_tf, default_target_tf = _resolve_loss(loss)
        lossfn = lossfn_resolved
        pred_tf = pred_tf === identity ? default_pred_tf : pred_tf
        target_tf = target_tf === identity ? default_target_tf : target_tf
    end

    # Optional TensorBoard logging. If `logger` is provided, it takes precedence.
    tb_logger = nothing
    cb = logger
    if cb === nothing && tb_logdir !== nothing && log_interval > 0
        tb_logger = TBLogger(tb_logdir)
        cb = (epoch::Int, l::Real) -> begin
            with_logger(tb_logger) do
                @info "train" epoch = epoch loss = Float32(l)
            end
            nothing
        end
    end

    # Flux.setup builds optimizer state for the given model parameters.
    opt_state = Flux.setup(opt_rule, m2)

    # Batch loss closure:
    # - Xb is (D, Tmax, B) padded minibatch
    # - lengths indicates valid Tᵢ for each batch element
    function loss_batch(mm, Xb, lengths)
        # Model produces reconstruction of same shape as Xb.
        Xhat = pred_tf(reconstruct_sequence(mm, Xb))
        Xt = target_tf(Xb)

        # Ensure loss sees (D, T, B).
        Xhat3 = _as3d(Xhat)
        Xt3 = _as3d(Xt)

        ℓ = lossfn(Xhat3, Xt3, lengths)

        # Optional L2 regularization on trainable arrays.
        if λ_reg != 0f0
            reg = 0f0
            for p in Flux.trainables(mm)
                reg += sum(abs2, p)
            end
            ℓ += λ_reg * reg
        end

        return ℓ
    end

    logged_losses = Float32[]
    do_log = log_interval > 0

    for epoch in 1:epochs
        # Generate padded minibatches each epoch.
        mb_iter = minibatches(
            Xs;
            batchsize = batchsize,
            shuffle = shuffle_each_epoch,
            rng = rng,
            pad_value = pad_value,
        )

        total = 0f0
        nb = 0

        for (Xb_cpu, lengths) in mb_iter
            # Move batch to device. 
            # lengths stay on CPU and is only used only for mask construction.
            Xb = _maybe_to_gpu_batch(Xb_cpu, device)

            # Compute loss and gradients w.r.t. model parameters.
            ℓ, grads = Flux.withgradient(m2) do mm
                loss_batch(mm, Xb, lengths)
            end

            # grads[1] corresponds to the gradient of the first argument 
            Flux.update!(opt_state, m2, grads[1])

            total += Float32(ℓ)
            nb += 1
        end

        avg = total / max(nb, 1)

        if do_log && (epoch % log_interval == 0)
            push!(logged_losses, avg)
            if cb !== nothing
                cb(epoch, avg)
            end
        end
    end

    Flux.testmode!(m2)

    if tb_logger !== nothing
        close(tb_logger)
    end

    # Return trained model on CPU if requested.
    m = _maybe_to_cpu_model(m2, return_cpu)

    return (; model = m, losses = logged_losses, device = device, Xs = Xs, lossname = lossname)
end