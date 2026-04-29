# src/clustering.jl
#
# Small clustering helpers  to enable evaluation of sequence-to-sequence autoencoding.

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
    silhouette_score(Z, assignments; metric=SqEuclidean()) -> Float32

Compute the mean silhouette score for an embedding matrix `Z`.

By convention, this package uses `Z` shaped `(d, N)` with columns as observations.
However, this function will also accept row-observation layout `(N, d)` if
`length(assignments) == size(Z, 1)`.

`assignments` must have one cluster label per observation.
"""
function silhouette_score(
    Z::AbstractMatrix,
    assignments::AbstractVector{<:Integer};
    metric = SqEuclidean(),
)
    n_rows = size(Z, 1)
    n_cols = size(Z, 2)

    # Decide whether observations are rows or columns based on assignment length.
    # Distances.pairwise(...; dims=2) treats columns as observations.
    obs_dims = if length(assignments) == n_cols
        2  # observations are columns (Z is d x N)
    elseif length(assignments) == n_rows
        1  # observations are rows    (Z is N x d)
    else
        throw(
            ArgumentError(
                "silhouette_score: length(assignments)=$(length(assignments)) must equal " *
                "size(Z,2)=$n_cols (columns-as-observations) or size(Z,1)=$n_rows (rows-as-observations)",
            ),
        )
    end

    N = obs_dims == 2 ? n_cols : n_rows

    # Pairwise distance matrix (N x N) across the observation dimension.
    Dmat = pairwise(metric, Z; dims = obs_dims)

    # Pre-index observations by cluster id indicating which observations belong to each cluster.
    clusters = unique(assignments)
    idxs_by = Dict(c => findall(==(c), assignments) for c in clusters)

    s = zeros(Float64, N)

    for i in 1:N
        ci = assignments[i]
        in_cluster = idxs_by[ci]

        # a(i): mean distance to other points in the same cluster.
        a = if length(in_cluster) <= 1
            0.0
        else
            mean(Dmat[i, j] for j in in_cluster if j != i)
        end

        # b(i): minimum mean distance to points in any other cluster.
        b = Inf
        for c in clusters
            c == ci && continue
            idxs = idxs_by[c]
            isempty(idxs) && continue
            b = min(b, mean(Dmat[i, j] for j in idxs))
        end

        denom = max(a, b)
        s[i] = denom == 0 ? 0.0 : (b - a) / denom
    end

    return Float32(mean(s))
end

# Internal helper: run kmeans so that returned assignments always align with the
# package convention "observations = columns of Z" (length(assignments) == size(Z,2)).
function _kmeans_columns(Z::AbstractMatrix, K::Int; rng = Random.default_rng())
    N = size(Z, 2)

    # Try as-is (Clustering.kmeans may treat columns as observations depending on version).
    res = kmeans(Z, K; rng = rng)
    if length(res.assignments) == N
        return res
    end

    # Try transposed (common if kmeans expects observations in rows).
    res2 = kmeans(permutedims(Z), K; rng = rng)
    if length(res2.assignments) == N
        return res2
    end

    throw(
        ArgumentError(
            "kmeans produced assignments lengths $(length(res.assignments)) and $(length(res2.assignments)); expected N=$N",
        ),
    )
end

"""
    best_kmeans_clustering(Z; Kmax=6, metric=SqEuclidean(), rng=Random.default_rng())
        -> (best_res, best_K, best_score)

Select `K ∈ 2:Kmax` using silhouette score.

Assumes `Z` is `(d, N)` with columns as observations, and returns:
- `best_res`: Clustering.jl kmeans result (with `best_res.assignments` length N)
- `best_K`: chosen number of clusters
- `best_score`: silhouette score for that clustering
"""
function best_kmeans_clustering(
    Z::AbstractMatrix;
    Kmax::Int = 6,
    metric = SqEuclidean(),
    rng = Random.default_rng(),
)
    best_res = nothing
    best_K = 0
    best_score = -Inf32

    for K in 2:Kmax
        res = _kmeans_columns(Z, K; rng = rng)
        score = silhouette_score(Z, res.assignments; metric = metric)

        if score > best_score
            best_score = score
            best_res = res
            best_K = K
        end
    end

    return (best_res, best_K, best_score)
end

"""
    cluster_composition(assignments, labels; K=nothing)

Count label occurrences per cluster.

- `assignments`: cluster id per observation
- `labels`: ground-truth label per observation (same length as `assignments`)
- `K`: optionally force number of clusters; otherwise uses `maximum(assignments)`

Returns: `Vector{Dict}` of length `K`, where element `k` maps `label => count`.
"""
function cluster_composition(
    assignments::AbstractVector{<:Integer},
    labels::AbstractVector;
    K::Union{Nothing,Int} = nothing,
)
    if length(assignments) != length(labels)
        throw(ArgumentError("cluster_composition: assignments and labels must have same length"))
    end

    K2 = isnothing(K) ? maximum(assignments) : K
    out = [Dict{eltype(labels),Int}() for _ in 1:K2]

    for (a, lbl) in zip(assignments, labels)
        d = out[a]
        d[lbl] = get(d, lbl, 0) + 1
    end

    return out
end

"""
    clustering_accuracy(assignments, labels) -> (acc, mapping)

Compute a simple clustering "accuracy" via majority-vote label per cluster.

- `mapping[k]` is the predicted label for cluster `k`.
- `acc` is the fraction of observations whose label matches their cluster's majority label.

Tie-break rule: if two labels are tied for majority within a cluster, choose the smaller label.
"""
function clustering_accuracy(
    assignments::AbstractVector{<:Integer},
    labels::AbstractVector{<:Integer},
)
    if length(assignments) != length(labels)
        throw(ArgumentError("clustering_accuracy: assignments and labels must have same length"))
    end

    K = maximum(assignments)
    mapping = Dict{Int,Int}()
    correct = 0

    for k in 1:K
        idxs = findall(==(k), assignments)
        isempty(idxs) && continue

        counts = Dict{Int,Int}()
        for i in idxs
            lbl = labels[i]
            counts[lbl] = get(counts, lbl, 0) + 1
        end

        maj_lbl = reduce(keys(counts)) do a, b
            ca, cb = counts[a], counts[b]
            (ca > cb || (ca == cb && a < b)) ? a : b
        end

        mapping[k] = maj_lbl
        for i in idxs
            correct += (labels[i] == maj_lbl) ? 1 : 0
        end
    end

    return (Float32(correct / length(labels)), mapping)
end
