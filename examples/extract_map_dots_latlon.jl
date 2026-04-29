module ExtractMapDotsLatLon

# ============================================================
# Purpose
# ============================================================
#
# This module extracts colored dots from a plain map image and
# estimates latitude/longitude for each detected dot.
#
# It is designed for the "Option B" workflow:
#   - you have a map image (PNG/JPG/etc.)
#   - you know several control points:
#       pixel x, pixel y, latitude, longitude
#   - you want to detect dots in the image
#   - you want to convert detected dot centroids into lat/lon
#
# The pipeline is:
#   1. Load image
#   2. Flatten any alpha/transparency onto a background color
#   3. Detect pixels close to a target color
#   4. Morphologically clean the binary mask
#   5. Find connected components and their centroids
#   6. Fit an affine pixel -> geographic transform from control points
#   7. Apply the transform to every detected dot
#   8. Write CSV and image diagnostics
#
# Important coordinate convention:
#   - image arrays are indexed as img[row, col] = img[y, x]
#   - throughout this module, we store:
#         x = column
#         y = row
#
# ============================================================

using FileIO
using Images
using Colors
using ImageMorphology
using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Dates

export ControlPoint,
       SamplePoint,
       DetectionConfig,
       AffineGeoTransform,
       read_control_points_csv,
       read_sample_points_csv,
       estimate_target_color_from_points,
       extract_dots_with_latlon,
       run_pipeline,
       save_mask_image,
       save_overlay_image

# ============================================================
# Data types
# ============================================================

"""
A control point linking a pixel location to a real geographic coordinate.

Fields:
- x   : pixel column
- y   : pixel row
- lat : latitude in decimal degrees
- lon : longitude in decimal degrees
"""
struct ControlPoint
    x::Float64
    y::Float64
    lat::Float64
    lon::Float64
end

"""
A sample point used only for estimating the target dot color.

Fields:
- x : pixel column
- y : pixel row
"""
struct SamplePoint
    x::Int
    y::Int
end

"""
Affine transform from image pixel coordinates (x, y)
to geographic coordinates (lat, lon):

    lat = a1*x + a2*y + a3
    lon = b1*x + b2*y + b3
"""
struct AffineGeoTransform
    lat_coeffs::Vector{Float64}   # length 3
    lon_coeffs::Vector{Float64}   # length 3
end

"""
Detection settings.

Fields:
- target_rgb       : RGB color that represents the dot color
- color_tolerance  : Euclidean distance threshold in RGB space
- min_pixels       : minimum connected component size to keep
- max_pixels       : maximum connected component size to keep
- erode_iters      : number of erosions applied to the raw mask
- dilate_iters     : number of dilations applied after erosion
- connectivity     : :four or :eight for connected components
- alpha_background : background color used if image has transparency
"""
Base.@kwdef struct DetectionConfig
    target_rgb::RGB{Float64} = RGB{Float64}(1.0, 0.0, 0.0)
    color_tolerance::Float64 = 0.20
    min_pixels::Int = 5
    max_pixels::Int = 10_000
    erode_iters::Int = 0
    dilate_iters::Int = 1
    connectivity::Symbol = :eight
    alpha_background::RGB{Float64} = RGB{Float64}(1.0, 1.0, 1.0)
end

# ============================================================
# General helpers
# ============================================================

"""
Normalize a CSV header so we can accept minor naming differences.
Examples:
- "Lat"        -> "lat"
- "pixel_x"    -> "pixelx"
- "Longitude " -> "longitude"
"""
function normalize_header(name)
    s = String(name)
    s = strip(lowercase(s))
    s = replace(s, "_" => "")
    s = replace(s, " " => "")
    return s
end

"""
Create the directory implied by an output prefix if it does not exist.

Example:
- prefix = "results/run1/map"
  creates directory "results/run1" if needed
"""
function ensure_output_dir(prefix::AbstractString)
    dir = dirname(prefix)
    if dir != "." && !isdir(dir)
        mkpath(dir)
    end
    return nothing
end

"""
Clamp an integer to [lo, hi].
"""
@inline clamp_int(v::Int, lo::Int, hi::Int) = max(lo, min(hi, v))

"""
Convert a real value to Int by rounding.
"""
@inline round_int(x::Real) = round(Int, x)

# ============================================================
# CSV readers
# ============================================================

"""
Read control points from CSV.

Accepted columns (case/spacing/underscore tolerant):
- x
- y
- lat or latitude
- lon or longitude
"""
function read_control_points_csv(path::AbstractString)
    df = CSV.read(path, DataFrame)

    name_map = Dict(normalize_header(n) => String(n) for n in names(df))

    function require_col(possible_names::Vector{String})
        for n in possible_names
            if haskey(name_map, n)
                return name_map[n]
            end
        end
        error("Missing required CSV column. Expected one of: $(join(possible_names, ", "))")
    end

    x_col   = require_col(["x"])
    y_col   = require_col(["y"])
    lat_col = require_col(["lat", "latitude"])
    lon_col = require_col(["lon", "longitude"])

    cps = ControlPoint[]
    for row in eachrow(df)
        x   = Float64(row[Symbol(x_col)])
        y   = Float64(row[Symbol(y_col)])
        lat = Float64(row[Symbol(lat_col)])
        lon = Float64(row[Symbol(lon_col)])

        push!(cps, ControlPoint(x, y, lat, lon))
    end

    validate_control_points(cps)
    return cps
end

"""
Read sample points from CSV.

Accepted columns:
- x
- y
"""
function read_sample_points_csv(path::AbstractString)
    df = CSV.read(path, DataFrame)

    name_map = Dict(normalize_header(n) => String(n) for n in names(df))

    function require_col(possible_names::Vector{String})
        for n in possible_names
            if haskey(name_map, n)
                return name_map[n]
            end
        end
        error("Missing required CSV column. Expected one of: $(join(possible_names, ", "))")
    end

    x_col = require_col(["x"])
    y_col = require_col(["y"])

    pts = SamplePoint[]
    for row in eachrow(df)
        x = round_int(Float64(row[Symbol(x_col)]))
        y = round_int(Float64(row[Symbol(y_col)]))
        push!(pts, SamplePoint(x, y))
    end

    isempty(pts) && error("Sample points CSV is empty.")
    return pts
end

# ============================================================
# Validation
# ============================================================

"""
Validate the control points before fitting.

Checks:
- at least 3 points
- all values finite
- no duplicate pixel points
- no duplicate lat/lon points
- affine design matrix has rank 3
- warn if the matrix is ill-conditioned
"""
function validate_control_points(cps::Vector{ControlPoint})
    n = length(cps)
    n < 3 && error("Need at least 3 control points.")

    all(isfinite(cp.x) && isfinite(cp.y) && isfinite(cp.lat) && isfinite(cp.lon) for cp in cps) ||
        error("All control point values must be finite.")

    pixel_pairs = [(cp.x, cp.y) for cp in cps]
    geo_pairs   = [(cp.lat, cp.lon) for cp in cps]

    length(unique(pixel_pairs)) == n || error("Duplicate pixel control points found.")
    length(unique(geo_pairs))   == n || error("Duplicate geographic control points found.")

    A = Matrix{Float64}(undef, n, 3)
    for (i, cp) in enumerate(cps)
        A[i, 1] = cp.x
        A[i, 2] = cp.y
        A[i, 3] = 1.0
    end

    rank(A) == 3 || error("Control points are collinear or otherwise degenerate for affine fitting.")

    κ = cond(A)
    if κ > 1e8
        @warn "Control points are very ill-conditioned; the affine fit may be unstable." condition_number=κ
    elseif κ > 1e5
        @warn "Control points are somewhat ill-conditioned; check leave-one-out errors carefully." condition_number=κ
    end

    return nothing
end

"""
Validate that sample points are inside the image bounds.
"""
function validate_sample_points(points::Vector{SamplePoint}, img::AbstractMatrix)
    h, w = size(img)
    for p in points
        (1 <= p.x <= w) || error("Sample point x=$(p.x) is outside image width 1:$w")
        (1 <= p.y <= h) || error("Sample point y=$(p.y) is outside image height 1:$h")
    end
    return nothing
end

# ============================================================
# Image loading and alpha handling
# ============================================================

"""
Flatten an RGBA image onto a solid background color.

Why this matters:
A PNG can contain transparency. Simply dropping alpha can give misleading
colors near edges. Flattening makes the RGB values predictable.
"""
function flatten_rgba_image(img_rgba::AbstractMatrix{<:RGBA};
                            background::RGB{Float64}=RGB{Float64}(1.0, 1.0, 1.0))
    h, w = size(img_rgba)
    out = Array{RGB{Float64}}(undef, h, w)

    @inbounds for y in 1:h
        for x in 1:w
            c = RGBA{Float64}(img_rgba[y, x])
            a = alpha(c)

            fg = RGB{Float64}(c)

            out[y, x] = RGB{Float64}(
                a * red(fg)   + (1 - a) * red(background),
                a * green(fg) + (1 - a) * green(background),
                a * blue(fg)  + (1 - a) * blue(background),
            )
        end
    end

    return out
end

"""
Load any image file and convert it to a flat RGB image.

We convert everything to RGBA first, then flatten onto a known background.
This handles RGB, RGBA, grayscale, and similar image types consistently.
"""
function load_rgb_image(path::AbstractString;
                        alpha_background::RGB{Float64}=RGB{Float64}(1.0, 1.0, 1.0))
    raw = load(path)
    rgba = RGBA{Float64}.(raw)
    rgb = flatten_rgba_image(rgba; background=alpha_background)
    return rgb
end

# ============================================================
# Color utilities
# ============================================================

"""
Euclidean distance in RGB space.

This is simple and easy to tune. It is not as perceptually sophisticated
as Lab-space distance, but it is usually good enough for map dots.
"""
@inline function rgb_distance(c1::RGB{T}, c2::RGB{S}) where {T<:Real,S<:Real}
    dr = red(c1)   - red(c2)
    dg = green(c1) - green(c2)
    db = blue(c1)  - blue(c2)
    return sqrt(dr*dr + dg*dg + db*db)
end

"""
Compute the mean RGB color in a small square patch centered on (x, y).

This is helpful when a dot is anti-aliased and its pixels are not all
exactly the same color.
"""
function patch_mean_rgb(img::AbstractMatrix{<:RGB}, x::Int, y::Int; radius::Int=2)
    h, w = size(img)
    x1 = clamp_int(x - radius, 1, w)
    x2 = clamp_int(x + radius, 1, w)
    y1 = clamp_int(y - radius, 1, h)
    y2 = clamp_int(y + radius, 1, h)

    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0
    n = 0

    @inbounds for yy in y1:y2
        for xx in x1:x2
            c = RGB{Float64}(img[yy, xx])
            sum_r += red(c)
            sum_g += green(c)
            sum_b += blue(c)
            n += 1
        end
    end

    n > 0 || error("Patch had no pixels, which should be impossible.")

    return RGB{Float64}(sum_r / n, sum_g / n, sum_b / n)
end

"""
Estimate the target dot color by averaging multiple sample patches.

Recommended workflow:
- place 3-10 sample points on known dots
- let this function estimate the average dot color
"""
function estimate_target_color_from_points(img::AbstractMatrix{<:RGB},
                                           points::Vector{SamplePoint};
                                           radius::Int=2)
    validate_sample_points(points, img)

    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0

    for p in points
        c = patch_mean_rgb(img, p.x, p.y; radius=radius)
        sum_r += red(c)
        sum_g += green(c)
        sum_b += blue(c)
    end

    n = length(points)
    return RGB{Float64}(sum_r / n, sum_g / n, sum_b / n)
end

# ============================================================
# Binary mask generation
# ============================================================

"""
Build a binary mask for pixels close to the target color.
"""
function detect_colored_pixels(img::AbstractMatrix{<:RGB}, cfg::DetectionConfig)
    h, w = size(img)
    mask = falses(h, w)

    target = cfg.target_rgb
    tol = cfg.color_tolerance

    @inbounds for y in 1:h
        for x in 1:w
            mask[y, x] = rgb_distance(img[y, x], target) <= tol
        end
    end

    return mask
end

"""
Apply erosion then dilation to clean the mask.

Typical uses:
- erosion removes isolated specks
- dilation reconnects slightly broken dot blobs
"""
function clean_mask(mask::BitMatrix, cfg::DetectionConfig)
    out = copy(mask)

    for _ in 1:cfg.erode_iters
        out = erode(out)
    end

    for _ in 1:cfg.dilate_iters
        out = dilate(out)
    end

    return out
end

"""
Return the connectivity kernel used by label_components.

- :four  => 4-neighbor connectivity
- :eight => 8-neighbor connectivity
"""
function connectivity_kernel(connectivity::Symbol)
    if connectivity == :four
        return Bool[
            false  true false
             true  true true
            false  true false
        ]
    elseif connectivity == :eight
        return trues(3, 3)
    else
        error("Unsupported connectivity: $connectivity. Use :four or :eight.")
    end
end

# ============================================================
# Connected component extraction
# ============================================================

"""
Extract connected components representing candidate dots.

Implementation notes:
- We use label_components from ImageMorphology.
- We then use component_lengths, component_indices, and component_centroids.
- component_centroids returns (row, col), which we convert to (y, x) then store as x, y.
- Bounding boxes are computed from the component indices.
"""
function extract_dot_centroids(mask::BitMatrix;
                               min_pixels::Int=5,
                               max_pixels::Int=10_000,
                               connectivity::Symbol=:eight)

    labels = label_components(mask, connectivity_kernel(connectivity))
    max_label = maximum(labels)

    counts = component_lengths(labels)
    inds   = component_indices(CartesianIndex, labels)
    cents  = component_centroids(labels)

    dot_id = Int[]
    xs = Float64[]
    ys = Float64[]
    areas = Int[]
    xmins = Int[]
    xmaxs = Int[]
    ymins = Int[]
    ymaxs = Int[]

    next_id = 1

    for lbl in 1:max_label
        area = counts[lbl]

        if area < min_pixels || area > max_pixels
            continue
        end

        # component_centroids returns (row, col) = (y, x)
        cy, cx = cents[lbl]

        comp_inds = inds[lbl]
        isempty(comp_inds) && continue

        xmin = typemax(Int)
        xmax = typemin(Int)
        ymin = typemax(Int)
        ymax = typemin(Int)

        for I in comp_inds
            y = I[1]
            x = I[2]
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
        end

        push!(dot_id, next_id)
        push!(xs, Float64(cx))
        push!(ys, Float64(cy))
        push!(areas, Int(area))
        push!(xmins, xmin)
        push!(xmaxs, xmax)
        push!(ymins, ymin)
        push!(ymaxs, ymax)

        next_id += 1
    end

    return DataFrame(
        dot_id = dot_id,
        x = xs,
        y = ys,
        area = areas,
        xmin = xmins,
        xmax = xmaxs,
        ymin = ymins,
        ymax = ymaxs,
    )
end

# ============================================================
# Affine transform fitting
# ============================================================

"""
Fit an affine transform from pixel coordinates to geographic coordinates.

Uses least squares when there are more than 3 control points.
"""
function fit_affine_transform(cps::Vector{ControlPoint})
    validate_control_points(cps)

    n = length(cps)
    A = Matrix{Float64}(undef, n, 3)
    lat_vec = Vector{Float64}(undef, n)
    lon_vec = Vector{Float64}(undef, n)

    for (i, cp) in enumerate(cps)
        A[i, 1] = cp.x
        A[i, 2] = cp.y
        A[i, 3] = 1.0

        lat_vec[i] = cp.lat
        lon_vec[i] = cp.lon
    end

    lat_coeffs = A \ lat_vec
    lon_coeffs = A \ lon_vec

    return AffineGeoTransform(lat_coeffs, lon_coeffs)
end

"""
Apply a fitted transform to one pixel coordinate.
"""
@inline function pixel_to_latlon(tform::AffineGeoTransform, x::Real, y::Real)
    lat = tform.lat_coeffs[1] * x + tform.lat_coeffs[2] * y + tform.lat_coeffs[3]
    lon = tform.lon_coeffs[1] * x + tform.lon_coeffs[2] * y + tform.lon_coeffs[3]
    return lat, lon
end

# ============================================================
# Error reporting
# ============================================================

"""
Approximate conversion of lat/lon error to meters.

This is a local approximation:
- latitude meters per degree is taken as about 111,320 m
- longitude meters per degree is scaled by cos(latitude)

Good enough for map-quality diagnostics.
"""
@inline function degree_error_to_meters(lat_err_deg::Real, lon_err_deg::Real, ref_lat_deg::Real)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * cosd(ref_lat_deg)
    return sqrt((lat_err_deg * m_per_deg_lat)^2 + (lon_err_deg * m_per_deg_lon)^2)
end

"""
Fit report on the same control points used to estimate the transform.

This is useful, but it can look deceptively good if you used exactly 3 points.
"""
function control_point_fit_report(tform::AffineGeoTransform, cps::Vector{ControlPoint})
    cp_id = Int[]
    xs = Float64[]
    ys = Float64[]
    true_lats = Float64[]
    true_lons = Float64[]
    pred_lats = Float64[]
    pred_lons = Float64[]
    lat_errors = Float64[]
    lon_errors = Float64[]
    euclidean_deg_error = Float64[]
    euclidean_meter_error = Float64[]

    for (i, cp) in enumerate(cps)
        pred_lat, pred_lon = pixel_to_latlon(tform, cp.x, cp.y)

        lat_err = pred_lat - cp.lat
        lon_err = pred_lon - cp.lon
        deg_err = sqrt(lat_err^2 + lon_err^2)
        meter_err = degree_error_to_meters(lat_err, lon_err, cp.lat)

        push!(cp_id, i)
        push!(xs, cp.x)
        push!(ys, cp.y)
        push!(true_lats, cp.lat)
        push!(true_lons, cp.lon)
        push!(pred_lats, pred_lat)
        push!(pred_lons, pred_lon)
        push!(lat_errors, lat_err)
        push!(lon_errors, lon_err)
        push!(euclidean_deg_error, deg_err)
        push!(euclidean_meter_error, meter_err)
    end

    return DataFrame(
        cp_id = cp_id,
        x = xs,
        y = ys,
        true_lat = true_lats,
        true_lon = true_lons,
        pred_lat = pred_lats,
        pred_lon = pred_lons,
        lat_error_deg = lat_errors,
        lon_error_deg = lon_errors,
        euclidean_deg_error = euclidean_deg_error,
        euclidean_meter_error = euclidean_meter_error,
    )
end

"""
Leave-one-out report.

For each control point:
- fit the transform using all the OTHER control points
- predict the held-out one
- report the prediction error

This is usually the most realistic quality check in this workflow.
"""
function leave_one_out_report(cps::Vector{ControlPoint})
    n = length(cps)

    if n < 4
        return DataFrame(
            cp_id = Int[],
            x = Float64[],
            y = Float64[],
            true_lat = Float64[],
            true_lon = Float64[],
            pred_lat = Float64[],
            pred_lon = Float64[],
            lat_error_deg = Float64[],
            lon_error_deg = Float64[],
            euclidean_deg_error = Float64[],
            euclidean_meter_error = Float64[],
        )
    end

    cp_id = Int[]
    xs = Float64[]
    ys = Float64[]
    true_lats = Float64[]
    true_lons = Float64[]
    pred_lats = Float64[]
    pred_lons = Float64[]
    lat_errors = Float64[]
    lon_errors = Float64[]
    euclidean_deg_error = Float64[]
    euclidean_meter_error = Float64[]

    for i in 1:n
        train = [cps[j] for j in 1:n if j != i]
        test = cps[i]

        tform = fit_affine_transform(train)
        pred_lat, pred_lon = pixel_to_latlon(tform, test.x, test.y)

        lat_err = pred_lat - test.lat
        lon_err = pred_lon - test.lon
        deg_err = sqrt(lat_err^2 + lon_err^2)
        meter_err = degree_error_to_meters(lat_err, lon_err, test.lat)

        push!(cp_id, i)
        push!(xs, test.x)
        push!(ys, test.y)
        push!(true_lats, test.lat)
        push!(true_lons, test.lon)
        push!(pred_lats, pred_lat)
        push!(pred_lons, pred_lon)
        push!(lat_errors, lat_err)
        push!(lon_errors, lon_err)
        push!(euclidean_deg_error, deg_err)
        push!(euclidean_meter_error, meter_err)
    end

    return DataFrame(
        cp_id = cp_id,
        x = xs,
        y = ys,
        true_lat = true_lats,
        true_lon = true_lons,
        pred_lat = pred_lats,
        pred_lon = pred_lons,
        lat_error_deg = lat_errors,
        lon_error_deg = lon_errors,
        euclidean_deg_error = euclidean_deg_error,
        euclidean_meter_error = euclidean_meter_error,
    )
end

"""
Return summary statistics for one numeric DataFrame column.
"""
function summarize_error(df::DataFrame, col::Symbol)
    if nrow(df) == 0
        return nothing
    end

    vals = collect(skipmissing(df[!, col]))
    isempty(vals) && return nothing

    return (
        mean = mean(vals),
        median = median(vals),
        max = maximum(vals),
    )
end

# ============================================================
# Applying the transform to detected dots
# ============================================================

"""
Add lat/lon columns to the detected-dot DataFrame.
"""
function apply_transform_to_dots(dots_df::DataFrame, tform::AffineGeoTransform)
    lats = Float64[]
    lons = Float64[]

    for row in eachrow(dots_df)
        lat, lon = pixel_to_latlon(tform, row.x, row.y)
        push!(lats, lat)
        push!(lons, lon)
    end

    out = copy(dots_df)
    out[!, :lat] = lats
    out[!, :lon] = lons
    return out
end

# ============================================================
# Visualization helpers
# ============================================================

"""
Save a binary mask as a black/white PNG.
"""
function save_mask_image(path::AbstractString, mask::BitMatrix)
    img = Gray{Float64}.(mask)
    save(path, img)
    return path
end

"""
Draw a simple cross marker in-place on an RGB image.
"""
function draw_cross!(img::AbstractMatrix{<:RGB}, x::Real, y::Real;
                     halfsize::Int=5,
                     color::RGB{Float64}=RGB{Float64}(0.0, 1.0, 0.0))
    h, w = size(img)
    xi = clamp_int(round_int(x), 1, w)
    yi = clamp_int(round_int(y), 1, h)

    for dx in -halfsize:halfsize
        xx = clamp_int(xi + dx, 1, w)
        img[yi, xx] = color
    end

    for dy in -halfsize:halfsize
        yy = clamp_int(yi + dy, 1, h)
        img[yy, xi] = color
    end

    return img
end

"""
Save an overlay image showing:
- detected dots in green
- control points in blue
- sample points in magenta
"""
function save_overlay_image(path::AbstractString,
                            base_img::AbstractMatrix{<:RGB},
                            dots_df::DataFrame;
                            control_points::Vector{ControlPoint}=ControlPoint[],
                            sample_points::Vector{SamplePoint}=SamplePoint[])
    img = copy(base_img)

    for row in eachrow(dots_df)
        draw_cross!(img, row.x, row.y; halfsize=4, color=RGB{Float64}(0.0, 1.0, 0.0))
    end

    for cp in control_points
        draw_cross!(img, cp.x, cp.y; halfsize=6, color=RGB{Float64}(0.0, 0.0, 1.0))
    end

    for p in sample_points
        draw_cross!(img, p.x, p.y; halfsize=6, color=RGB{Float64}(1.0, 0.0, 1.0))
    end

    save(path, img)
    return path
end

# ============================================================
# Metadata / output helpers
# ============================================================

"""
Write a small plain-text summary of the run so the results are reproducible later.
"""
function write_run_metadata(path::AbstractString;
                            image_path::AbstractString,
                            control_points_csv::AbstractString,
                            sample_points_csv::Union{Nothing,AbstractString},
                            cfg::DetectionConfig,
                            estimated_target::RGB{Float64},
                            result)
    open(path, "w") do io
        println(io, "run_timestamp = ", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
        println(io, "image_path = ", image_path)
        println(io, "control_points_csv = ", control_points_csv)
        println(io, "sample_points_csv = ", isnothing(sample_points_csv) ? "none" : sample_points_csv)

        println(io, "target_rgb = (",
            red(estimated_target), ", ",
            green(estimated_target), ", ",
            blue(estimated_target), ")")

        println(io, "color_tolerance = ", cfg.color_tolerance)
        println(io, "min_pixels = ", cfg.min_pixels)
        println(io, "max_pixels = ", cfg.max_pixels)
        println(io, "erode_iters = ", cfg.erode_iters)
        println(io, "dilate_iters = ", cfg.dilate_iters)
        println(io, "connectivity = ", cfg.connectivity)
        println(io, "alpha_background = (",
            red(cfg.alpha_background), ", ",
            green(cfg.alpha_background), ", ",
            blue(cfg.alpha_background), ")")

        println(io, "detected_dot_count = ", nrow(result.dots_geo_df))
    end

    return path
end

"""
Write a DataFrame to CSV and return the output path.
"""
function write_dataframe_csv(path::AbstractString, df::DataFrame)
    CSV.write(path, df)
    return path
end

# ============================================================
# Main extraction pipeline
# ============================================================

"""
Core extraction function.

Inputs:
- image_path      : path to the image file
- control_points  : vector of ControlPoint
- cfg             : detection configuration

Returns a named tuple containing:
- image
- raw_mask
- clean_mask
- dots_pixels_df
- dots_geo_df
- transform
- control_fit_df
- loo_df
"""
function extract_dots_with_latlon(image_path::AbstractString,
                                  control_points::Vector{ControlPoint};
                                  cfg::DetectionConfig=DetectionConfig())

    img = load_rgb_image(image_path; alpha_background=cfg.alpha_background)

    raw_mask = detect_colored_pixels(img, cfg)
    cleaned = clean_mask(raw_mask, cfg)

    dots_pixels_df = extract_dot_centroids(
        cleaned;
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
        connectivity=cfg.connectivity,
    )

    tform = fit_affine_transform(control_points)
    dots_geo_df = apply_transform_to_dots(dots_pixels_df, tform)

    control_fit_df = control_point_fit_report(tform, control_points)
    loo_df = leave_one_out_report(control_points)

    return (
        image = img,
        raw_mask = raw_mask,
        clean_mask = cleaned,
        dots_pixels_df = dots_pixels_df,
        dots_geo_df = dots_geo_df,
        transform = tform,
        control_fit_df = control_fit_df,
        loo_df = loo_df,
    )
end

# ============================================================
# One-call convenience runner
# ============================================================

"""
High-level convenience function.

Workflow:
1. Read control points CSV
2. Load image
3. Determine target color:
   - if sample_points_csv is provided, estimate color from those points
   - otherwise use manual target_r, target_g, target_b
4. Run extraction
5. Write all outputs

Recommended usage:
- provide 4-10 control points
- provide 3-10 sample points on known dots
- inspect the overlay and clean mask before trusting the lat/lon output
"""
function run_pipeline(; image_path::AbstractString,
                         control_points_csv::AbstractString,
                         output_prefix::AbstractString="output/result",
                         sample_points_csv::Union{Nothing,AbstractString}=nothing,
                         sample_radius::Int=2,
                         target_r::Float64=1.0,
                         target_g::Float64=0.0,
                         target_b::Float64=0.0,
                         color_tolerance::Float64=0.20,
                         min_pixels::Int=5,
                         max_pixels::Int=10_000,
                         erode_iters::Int=0,
                         dilate_iters::Int=1,
                         connectivity::Symbol=:eight,
                         alpha_bg_r::Float64=1.0,
                         alpha_bg_g::Float64=1.0,
                         alpha_bg_b::Float64=1.0)

    ensure_output_dir(output_prefix)

    control_points = read_control_points_csv(control_points_csv)

    # Load once here so we can estimate target color if sample points were provided.
    alpha_bg = RGB{Float64}(alpha_bg_r, alpha_bg_g, alpha_bg_b)
    base_img = load_rgb_image(image_path; alpha_background=alpha_bg)

    sample_points = SamplePoint[]
    target_rgb = RGB{Float64}(target_r, target_g, target_b)

    if !isnothing(sample_points_csv)
        sample_points = read_sample_points_csv(sample_points_csv)
        validate_sample_points(sample_points, base_img)
        target_rgb = estimate_target_color_from_points(base_img, sample_points; radius=sample_radius)
    end

    cfg = DetectionConfig(
        target_rgb = target_rgb,
        color_tolerance = color_tolerance,
        min_pixels = min_pixels,
        max_pixels = max_pixels,
        erode_iters = erode_iters,
        dilate_iters = dilate_iters,
        connectivity = connectivity,
        alpha_background = alpha_bg,
    )

    result = extract_dots_with_latlon(image_path, control_points; cfg=cfg)

    # Write tabular outputs
    write_dataframe_csv(output_prefix * "_dots_pixels.csv", result.dots_pixels_df)
    write_dataframe_csv(output_prefix * "_dots_latlon.csv", result.dots_geo_df)
    write_dataframe_csv(output_prefix * "_control_fit.csv", result.control_fit_df)
    write_dataframe_csv(output_prefix * "_leave_one_out.csv", result.loo_df)

    # Write image diagnostics
    save_mask_image(output_prefix * "_raw_mask.png", result.raw_mask)
    save_mask_image(output_prefix * "_clean_mask.png", result.clean_mask)
    save_overlay_image(
        output_prefix * "_overlay.png",
        result.image,
        result.dots_pixels_df;
        control_points=control_points,
        sample_points=sample_points,
    )

    # Write metadata
    write_run_metadata(
        output_prefix * "_run_metadata.txt";
        image_path=image_path,
        control_points_csv=control_points_csv,
        sample_points_csv=sample_points_csv,
        cfg=cfg,
        estimated_target=target_rgb,
        result=result,
    )

    # Console summary
    println("Done.")
    println("Detected dots: ", nrow(result.dots_geo_df))
    println("Target RGB used: (", red(target_rgb), ", ", green(target_rgb), ", ", blue(target_rgb), ")")

    fit_summary = summarize_error(result.control_fit_df, :euclidean_meter_error)
    if fit_summary !== nothing
        println("Control-fit error (meters):")
        println("  mean   = ", fit_summary.mean)
        println("  median = ", fit_summary.median)
        println("  max    = ", fit_summary.max)
    end

    if nrow(result.loo_df) > 0
        loo_summary = summarize_error(result.loo_df, :euclidean_meter_error)
        println("Leave-one-out error (meters):")
        println("  mean   = ", loo_summary.mean)
        println("  median = ", loo_summary.median)
        println("  max    = ", loo_summary.max)
    else
        println("Leave-one-out report not generated because fewer than 4 control points were supplied.")
    end

    println("Wrote:")
    println("  ", output_prefix * "_dots_pixels.csv")
    println("  ", output_prefix * "_dots_latlon.csv")
    println("  ", output_prefix * "_control_fit.csv")
    println("  ", output_prefix * "_leave_one_out.csv")
    println("  ", output_prefix * "_raw_mask.png")
    println("  ", output_prefix * "_clean_mask.png")
    println("  ", output_prefix * "_overlay.png")
    println("  ", output_prefix * "_run_metadata.txt")

    return result
end

end # module