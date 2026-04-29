# ./test/runtests.jl
# Package test entry point that validates shapes, batching and masking, training, and 
# clustering utilities.

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

using Test
using Random
using Optimisers
using TransformerAutoencoding

# Package-level test set
@testset "TransformerAutoencoding" begin
    Random.seed!(0)  # stable randomness across runs/CI

    # Small fixed toy input (unbatched)
    input_dim = 4
    T = 7
    X = rand(Float32, input_dim, T)

    # Small model for tests
    m = build_autoencoder(input_dim; d_model = 16, num_layers = 1, num_heads = 4, dropout = 0.0)

    @testset "shapes" begin
        # Reconstruct returns the same shape as the input.
        Xhat = reconstruct_sequence(m, X)
        @test size(Xhat) == size(X)

        # Encoding a single sequence returns a length-d_model vector.
        z = encode_sequence(m, X)
        @test length(z) == 16

        # Decoding an embedding returns the original (D, T) shape.
        Xdec = decode_embedding(m, z, T)
        @test size(Xdec) == size(X)
    end

    @testset "training runs (CPU)" begin
        # Small dataset for a test of the training loop.
        Xs = [rand(Float32, input_dim, T) for _ in 1:20]

        # Baseline reconstruction error before training.
        mse0, _, _ = reconstruction_mse_stats(m, Xs)

        # Train for a few epochs on CPU. Use a fixed RNG for stable minibatch order.
        fit = train_autoencoder!(
            m,
            Xs;
            epochs = 5,
            opt_rule = Optimisers.Descent(1e-2),
            log_interval = 1,
            use_gpu = false,      # keep unit tests CPU-only
            return_cpu = true,    # ensure returned model is CPU arrays
            rng = MersenneTwister(1),
        )

        @test fit.device == :cpu
        @test !isempty(fit.losses)  # logging occurred

        # Post-training reconstruction error should be finite and not blow up.
        mse1, _, _ = reconstruction_mse_stats(fit.model, Xs)
        @test isfinite(mse1)
        # Not requiring strict decrease (SGD noise), just sanity.
        @test mse1 < 10f0 * mse0
    end

    @testset "clustering utilities" begin
        # Two well-separated 2D Gaussians 
        Z1 = randn(Float32, 2, 20) .+ [-3f0; 0f0]
        Z2 = randn(Float32, 2, 20) .+ [ 3f0; 0f0]
        Z = hcat(Z1, Z2)  # (d, N) with columns as observations

        assignments = vcat(fill(1, 20), fill(2, 20))

        # should return a silhouette score in [-1, 1].
        s = silhouette_score(Z, assignments)
        @test -1f0 <= s <= 1f0
    end

    @testset "batching + masking" begin
        rng = MersenneTwister(42)
        D = 4

        # Variable-length sequences (D, Tᵢ)
        Xs = [
            rand(rng, Float32, D, 5),
            rand(rng, Float32, D, 7),
            rand(rng, Float32, D, 3),
        ]

        # Pad into a single (D, Tmax, B) array and track true lengths.
        pad_value = -1f0
        Xpad, lengths = pad_sequences(Xs; pad_value = pad_value)

        @test size(Xpad, 1) == D
        @test size(Xpad, 3) == length(Xs)
        @test lengths == [5, 7, 3]
        @test size(Xpad, 2) == 7  # Tmax

        # Padded regions should equal pad_value.
        @test all(Xpad[:, 6:7, 1] .== pad_value)  # seq1 padded
        @test all(Xpad[:, 4:7, 3] .== pad_value)  # seq3 padded

        # minibatches yields padded batches + corresponding lengths.
        mb = collect(minibatches(Xs; batchsize = 2, shuffle = false, rng = rng, pad_value = pad_value))
        @test length(mb) == 2
        Xb1, l1 = mb[1]
        @test size(Xb1, 1) == D
        @test size(Xb1, 3) == 2
        @test l1 == [5, 7]

        # Masked loss should ignore padding timesteps.
        # If we modify only padded entries, masked loss should remain ~0.
        Xhat = copy(Xpad)
        Xhat[:, 6:7, 1] .+= 10f0
        Xhat[:, 4:7, 3] .-= 10f0
        @test masked_mse_loss(Xhat, Xpad, lengths) ≤ 1f-6
    end

    @testset "batched encode/reconstruct" begin
        rng = MersenneTwister(7)
        D = 4
        Xs = [rand(rng, Float32, D, t) for t in (5, 7, 6)]
        Xpad, lengths = pad_sequences(Xs; pad_value = 0f0)

        # Batched reconstruction: input/output should be (D, Tmax, B).
        Xhat = reconstruct_sequence(m, Xpad)
        @test size(Xhat) == size(Xpad)

        # Batched encoding returns (d_model, B).
        Zb = encode_sequence(m, Xpad)
        @test size(Zb, 1) == 16
        @test size(Zb, 2) == length(lengths)
    end

    @testset "best_kmeans_clustering" begin
        rng = MersenneTwister(9)
        Z1 = randn(rng, Float32, 2, 20) .+ [-3f0; 0f0]
        Z2 = randn(rng, Float32, 2, 20) .+ [ 3f0; 0f0]
        Z  = hcat(Z1, Z2)  # (d, N)

        # Should return a result with assignments of length N and finite score.
        res, K, s = best_kmeans_clustering(Z; Kmax = 4, rng = rng)
        @test 2 <= K <= 4
        @test length(res.assignments) == size(Z, 2)
        @test isfinite(s)
    end
end
