<p align="center">
  <img src=".github/transformerautoencoding.jl%20graphic.png 2" alt="TransformerAutoencoding.jl overview" width="800">
</p>

# TransformerAutoencoding.jl

`TransformerAutoencoding.jl` provides sequence-to-sequence autoencoding utilities for variable-length sequence datasets in Julia, using lightweight Transformer blocks implemented in Flux. It supports training on padded minibatches with masked losses so that padding does not affect optimization.
Common uses include preprocessing time series data for clustering, information retrieval, anomaly detection, and feature extraction for downstream models.

## Installation

From the repository root:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
````

Then:

```julia
using TransformerAutoencoding
```

## Minimum Working Example

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()

using TransformerAutoencoding
using Random
using Optimisers

# Synthetic variable-length dataset: N sequences of shape (D, Tᵢ)
Random.seed!(0)
D = 4
Xs = [rand(Float32, D, T) for T in rand(8:20, 64)]

# Build a small Transformer autoencoder
m = build_autoencoder(D; d_model=32, num_layers=2, num_heads=4, dropout=0.1)

# Train (CPU by default)
fit = train_autoencoder!(
    m, Xs;
    epochs=10,
    batchsize=32,
    opt_rule=Optimisers.Adam(1e-3),
    pad_value=0f0,
    lossfn=masked_mse_loss,
    log_interval=5,
    tb_logdir=nothing,
    use_gpu=false,
    return_cpu=true,
)

# Embed sequences for clustering / retrieval
Z = embed_sequences(fit.model, Xs; batchsize=64)  # (d_model, N)

# Unsupervised validation example
res, K, s = best_kmeans_clustering(Z; Kmax=6)
println((K=K, silhouette=s, device=fit.device))
```

## Data Format and Preparation

The primary dataset format is:

* `Xs::Vector{<:AbstractMatrix}` 
* Each `X` in `Xs` is shaped `(D, T_i)`
* `D` (feature dimension) is constant across sequences; `T_i` may vary

To convert a vector of time-series data into the correct input format for TransformerAutoencoding.jl, the user can utilize the following function:

```julia
Xs, μ, σ = prepare_sequence_dataset(seqs; normalize=true)
```

Supported inputs for seqs in `prepare_sequence_dataset` are:

* `Vector{<:AbstractMatrix}` where each matrix is already sized `(D, T)`
* `Vector{<:Vector{<:AbstractVector}}` where each sequence element is a timestep vector of length `D`

Returned values:

* `Xs`: `Vector{Matrix{Float32}}` in `(D, T_i)` form
* `μ`, `σ`: feature-wise normalization statistics (`Float32` vectors of length `D`)


### Variable-length Batching Utilities

Note that TransformerAutoencoding.jl offers utilities for padding sequences to enable a efficient tensor-based implementation, while leveraging masked loss functions to ensure that padded values do not impact training outcomes. 
The following utilities  are available for direct use and for integration into custom training loops with variable-length sequence data:

* `pad_sequences(Xs; pad_value=0f0, T_max=nothing)` -> `(Xpad, lengths)`

  * `Xpad` has shape `(D, Tmax, B)` with `pad_value` entry padding for sequences shorter than the maximum sequence length. 
  * `lengths` is a `Vector{Int}` of length `B` with `lengths[i] = length(Xs[i])`.

* `minibatches(Xs; batchsize=32, shuffle=true, rng=..., pad_value=0f0)` yields `(Xpad, lengths)` per minibatch

## Model Construction

A Transformer-based sequence-to-sequence autoencoder is constructed with:

```julia
input_dim = size(Xs[1], 1)

m = build_autoencoder(
    input_dim;
    d_model = 64,
    num_layers = 2,
    num_heads = 4,
    dropout = 0.1,
)
```

Core model functions:

* `encode_sequence(m, X)` returns the embedding of the model `m` for either a single sequence or a batch of sequences `X`.
* `reconstruct_sequence(m, X)` returns the reconstructed sequence or sequences using model `m`  and single sequence or batch of sequences `X`. 
* `decode_embedding(m, z, T)` returns the reconstruction for a single embedding vector `z` and sequence length `T`.

These functions can leverage batching using the optional `; batchsize=b` key word argument.

## Training

Training is performed with a single entry point `train_autoencoder!`:

```julia
using Optimisers

fit = train_autoencoder!(
    m,
    Xs;
    epochs = 20,
    batchsize = 64,
    opt_rule = Optimisers.Adam(1e-3),
    pad_value = 0f0,

    # choose a loss function (or pass a custom loss function)
    lossfn = masked_mse_loss,

    # optional logging
    log_interval = 10,
    tb_logdir = nothing,

    # optional device control
    use_gpu = false,
    return_cpu = true,
)
```

Returned fields include:

* `fit.model`: trained model (optionally moved back to CPU).
* `fit.losses`: logged losses at the chosen interval.
* `fit.device`: `:cpu` or `:cuda`, indicating which device the training was performed.
* `fit.lossname`: `:custom` if `lossfn` was provided, otherwise the resolved `Symbol`.

### Loss Functions


TransformerAutoencoding.jl contains several built-in loss functions for common losses: mean squared error (`lossfn = :mse`), binary cross entropy (`lossfn = :bce`), and logit binary cross entropy (`lossfn = :logitbce`). 
Importantly, these loss functions use masked loss functions to ensure padding does not affect training. 

TransformerAutoencoding.jl also allows for arbitrary differentiable loss functions following the following format:
```julia
lossfn(Xhat3, X3, lengths) -> scalar
```

Where:

* `Xhat3` (reconstructed batch) and `X3` (original batch) both have shape `(D, T, B)`.
* `lengths` is a `Vector{Int}` of length `B` indicating valid lengths per sequence in the minibatch.

Users may  train a transformer-based autoencoder using a custom loss function using `TransformerAutoencoding.jl` as follows: 
```julia
my_loss = (Xhat3, X3, lengths) -> masked_mse_loss(Xhat3, X3, lengths) + 1f-4 * sum(abs2, Xhat3)
fit = train_autoencoder!(m, Xs; lossfn=my_loss)
```  

## CUDA GPU Support

CUDA acceleration is enabled through the `use_gpu `flag in `train_autoencoder!`. CUDA availability is determined by the active environment configuration.

Typical usage:

```julia
fit = train_autoencoder!(m, Xs; use_gpu=true, lossfn=masked_mse_loss)
println(fit.device)  # :cuda or :cpu
```

If CUDA is not installed or not functional, set `use_gpu=false` or install `CUDA.jl` in the active environment:

```julia
import Pkg
Pkg.add("CUDA")
```

## TensorBoard Logging

Training can log scalars to `TensorBoardLogger.jl` when `tb_logdir` is set and `log_interval > 0`:

```julia
fit = train_autoencoder!(
    m, Xs;
    lossfn = masked_mse_loss,
    log_interval = 10,
    tb_logdir = "runs/experiment_1",
)
```

`TensorBoardLogger.jl` must be installed in the active environment.

## Examples

From the repository root:

```julia
include("examples/toy_gridworld.jl")
include("examples/grid_toy_gridworld_min.jl")
```

These examples cover dataset preparation, training, embedding, and clustering validation.

## Distribution Statement

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of War for Research and Engineering under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of War for Research and Engineering.

© 2026 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
