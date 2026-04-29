#   ./src/precompile.jl
# PrecompileTools workloads that precompile common model/training paths 
# to reduce first-run latency.

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

@setup_workload begin
    # PrecompileTools workload:
    # - runs at package precompile time to cache method specializations
    # - keeps an intentionally tiny problem size to reduce precompile time
    # - CPU-only by design (GPU availability varies across environments)

    # Small synthetic dataset: 8 sequences, each shaped (D=4, T=8)
    Xs = [rand(Float32, 4, 8) for _ in 1:8]

    # Small model used only to force compilation of common hot paths
    m = build_autoencoder(4; d_model = 16, num_layers = 1, num_heads = 4, dropout = 0.0)

    @compile_workload begin
        # Core user-facing calls that should be fast on first use
        encode_sequence(m, Xs[1])                 # single-sequence embedding
        reconstruct_sequence(m, Xs[1])            # single-sequence reconstruction
        embed_sequences(m, Xs; batchsize = 8)     # batched embedding path (padding + minibatches)

        # Loss function specialization (3D tensors + lengths vector)
        X3 = reshape(Xs[1], 4, 8, 1)              # (D, T, B=1)
        masked_mse_loss(X3, X3, [8])              # lengths=[T] for a single sequence
    end
end
