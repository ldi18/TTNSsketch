include(joinpath(@__DIR__, "..", "..", "TTNSsketch.jl", "src", "TTNSsketch.jl"))
using .TTNSsketch
using .TTNSsketch.ExampleTopologies
using .TTNSsketch.GraphicalModels
using .TTNSsketch.CoreDeterminingEquations
using .TTNSsketch.ErrorReporting: report_errors
using .TTNSsketch.Sketching
using Printf
using CSV
using DataFrames
using Plots
using LaTeXStrings
using Random
using Statistics
using BenchmarkTools
using ITensors: set_warn_order

"""
Create sketching kwargs for Perturbative sketching with given beta_dim.
When use_expansion=false, the order parameter is ignored and set to n_vertices-1 internally.
Matches full_workflow.jl exactly.
"""
function create_sketching_kwargs(beta_dim::Int, seed::Int)
  return Dict{Symbol, Any}(
    :sketching_type => Sketching.Perturbative,
    # Note: order is not set here - when use_expansion=false, it's automatically set to n_vertices-1
    :seed => seed,
    :epsilon => 1.0,
    :beta_dim => beta_dim,
    :use_expansion => false,
  )
end

"""
Run a single benchmark for given orders and beta_dim and return summary statistics.
Returns `(sum_ground_truth, sum_recov, mean_rel_error, std_rel_error, ttns_recov, p_ground_truth, runtime_seconds, runtime_dict)`.
If `track_runtime=true`, `runtime_dict` contains per-site runtime information.
"""
function benchmark_single(order::Int, beta_dim::Int;
                          n_vertices::Int = 6,
                          β::Real = 1.0,
                          seed::Int = 1234,
                          track_runtime::Bool = false)
  # Pre-create topology and ground truth outside timing
  ttns = ExampleTopologies.Linear(n_vertices)
  d = length(ttns.x_indices)
  set_warn_order(2*d+1)  # Match full_workflow.jl
  p_ground_truth = TTNSsketch.higher_order_probability_dict(ttns; β=β, order=order)
  sketching_kwargs = create_sketching_kwargs(beta_dim, seed)
  TTNSsketch.Sketching.Perturbative.RNG[] = nothing
  ttns_recov = deepcopy(ttns)

  bench_result = @benchmark CoreDeterminingEquations.compute_Gks!(
    $p_ground_truth, $ttns_recov; 
    sketching_kwargs=$sketching_kwargs
  ) evals=1 seconds=0.1  # seconds parameter ensures enough samples after warm-up
  
  # ttns_recov is already modified in place by compute_Gks!, so we can use it directly
  runtime_seconds = mean(bench_result.times) / 1e9  # Convert nanoseconds to seconds
  runtime_dict = track_runtime ? nothing : nothing  # Not tracking per-site runtime for now

  keys_all = collect(keys(p_ground_truth))
  sum_ground_truth = sum(values(p_ground_truth))
  sum_recov = sum(TTNSsketch.evaluate(ttns_recov, key) for key in keys_all)
  
  error_stats = report_errors(p_ground_truth, ttns_recov, keys_all; print_sections=false)
  mean_rel_error = error_stats.overall.mean
  mean_rel_error = isnan(mean_rel_error) ? 0.0 : mean_rel_error
  std_rel_error = error_stats.overall.std  # This is std across points, which is what we want for error bars
  std_rel_error = isnan(std_rel_error) ? 0.0 : std_rel_error
  return sum_ground_truth, sum_recov, mean_rel_error, std_rel_error, ttns_recov, p_ground_truth, runtime_seconds, runtime_dict
end

"""
Create error plot object (for combined plots).
"""
function create_error_plot(df, sketching_label; threshold=1, high_compression=0.1, fontsize=18)
  # Check if DataFrame is empty
  if nrow(df) == 0
    error("DataFrame is empty - no data to plot")
  end
  
  # Check if we have the required columns
  if !(:mean_rel_error in propertynames(df))
    error("DataFrame missing 'mean_rel_error' column")
  end
  
  # Handle missing error bars - use std_rel_error_mean if mean_rel_error_std is missing/NaN
  if !(:mean_rel_error_std in propertynames(df))
    df[!, :mean_rel_error_std] = zeros(nrow(df))
  end
  
  # Replace NaN values in mean_rel_error_std with 0
  # mean_rel_error_std represents the std of pointwise errors (variance across points, not runs)
  df.mean_rel_error_std = [ismissing(x) || (typeof(x) <: Number && isnan(x)) ? 0.0 : x for x in df.mean_rel_error_std]
  
  # If we have old data with std_rel_error_mean, use it as fallback (old format had it in a different column)
  if :std_rel_error_mean in propertynames(df)
    for i in 1:nrow(df)
      if (df.mean_rel_error_std[i] == 0.0 || isnan(df.mean_rel_error_std[i])) && 
         !ismissing(df.std_rel_error_mean[i]) && !isnan(df.std_rel_error_mean[i])
        df.mean_rel_error_std[i] = df.std_rel_error_mean[i]
      end
    end
  end
  
  # Use mean_rel_error instead of max_rel_error
  error_upper = df.mean_rel_error .+ df.mean_rel_error_std
  finite_errors = filter(isfinite, error_upper)
  if isempty(finite_errors)
    finite_errors = filter(isfinite, df.mean_rel_error)
    if isempty(finite_errors)
      error("No finite error values found in DataFrame")
    end
    max_error = maximum(finite_errors)
  else
    max_error = maximum(finite_errors)
  end
  
  transformed_errors = [scale_break_transform(y, max_error, threshold, high_compression) for y in df.mean_rel_error]
  transformed_error_bars = Float64[]
  for (y, std) in zip(df.mean_rel_error, df.mean_rel_error_std)
    if std > 0 && isfinite(std)
      # Transform the upper bound and take difference
      y_upper = y + std
      y_transformed = scale_break_transform(y, max_error, threshold, high_compression)
      y_upper_transformed = scale_break_transform(y_upper, max_error, threshold, high_compression)
      push!(transformed_error_bars, max(y_upper_transformed - y_transformed, 1e-10))
    else
      push!(transformed_error_bars, 0.0)
    end
  end
  df_transformed = copy(df)
  df_transformed[!, :transformed_error] = transformed_errors
  df_transformed[!, :transformed_error_bar] = transformed_error_bars
  
  # Use LaTeX labels for proper rendering of formulas
  yticks_pos, yticks_labels = generate_yticks(transformed_errors, df.mean_rel_error, threshold, high_compression; use_latex=true)
  
  # Calculate y-axis limits
  min_y = minimum(transformed_errors)
  max_y = maximum(transformed_errors)
  y_range = max_y - min_y
  ylims_bottom = min_y - 0.02 * y_range
  ylims_top = max_y + 0.05 * y_range
  
  # Define consistent color palette
  order_vals = sort(unique(df.order))
  
  # Generate x-axis ticks with LaTeX strings
  beta_dim_vals = sort(unique(df.beta_dim))
  xticks_pos = Float64.(beta_dim_vals)
  xticks_labels = [latexstring("$val") for val in beta_dim_vals]
  
  plt = plot(xlabel=latexstring("\\beta_{\\mathrm{dim}}"), 
             ylabel=latexstring("\\mathrm{Relative~error}"),
             title=latexstring("\\mathrm{(a)~Perturbative~Sketching:~Rel.~error~vs.~}\\beta_{\\mathrm{dim}}"),
             legend=false,
             xticks=(xticks_pos, xticks_labels),
             yticks=(yticks_pos, yticks_labels),
             ylims=(ylims_bottom, ylims_top),
             fontsize=fontsize, tickfontsize=fontsize,
             guidefontsize=fontsize, legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm,
             top_margin=14Plots.mm,
             linewidth=2.5, markersize=7, markerstrokewidth=0.1)
  
  for (idx, order_val) in enumerate(order_vals)
    sub = df_transformed[df_transformed.order .== order_val, :]
    if nrow(sub) == 0
      continue  # Skip if no data for this order
    end
    idx_sorted = sortperm(sub.beta_dim)
    x_vals = sub.beta_dim[idx_sorted]
    y_vals = sub.transformed_error[idx_sorted]
    y_err = sub.transformed_error_bar[idx_sorted]
    
    # Filter out NaN and Inf values
    valid_idx = [i for i in 1:length(y_vals) if isfinite(y_vals[i]) && isfinite(y_err[i])]
    if length(valid_idx) == 0
      continue  # Skip if no valid data points
    end
    
    x_vals_clean = x_vals[valid_idx]
    y_vals_clean = y_vals[valid_idx]
    y_err_clean = y_err[valid_idx]
    
    # Ensure error bars are positive and visible (minimum 1% of y value for visibility)
    y_err_clean = [max(err, abs(y) * 0.01, 1e-10) for (err, y) in zip(y_err_clean, y_vals_clean)]
    
    # Always plot with error bars
    plot!(plt, x_vals_clean, y_vals_clean, yerror=y_err_clean,
          marker=:o, label=latexstring("n_{\\mathrm{Cond.}} = $(order_val)"), 
          color=idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1,
          capsize=3)  # Add cap size to make error bars more visible
  end
  
  return plt
end

"""
Create runtime plot object (for combined plots).
Shows runtime vs beta_dim as a line plot with connected dots.
"""
function create_runtime_plot(df, sketching_label; n_vertices::Int, β::Real, fontsize=18)
  # Check if runtime data exists
  if !(:runtime_mean in propertynames(df)) || all(ismissing.(df.runtime_mean))
    println("WARNING: No runtime data available - skipping runtime plot")
    return nothing
  end
  
  all_runtimes = df.runtime_mean[.!ismissing.(df.runtime_mean)]
  if isempty(all_runtimes)
    println("WARNING: No valid runtime data - skipping runtime plot")
    return nothing
  end
  
  min_runtime = minimum(all_runtimes)
  max_runtime = maximum(all_runtimes)
  
  # Set y-axis limits to fixed range [10^-1 to 1.6 * 10^-1]
  ylims_bottom = 1e-1
  ylims_top = 1.6e-1
  
  # Generate y-axis ticks with LaTeX formatting (similar to error plot)
  # For runtime, format values in scientific notation with LaTeX, but without "e" and "+00"
  n_ticks = 5
  yticks_pos = collect(range(ylims_bottom, ylims_top, length=5))
  
  yticks_labels = Any[]
  for v in yticks_pos
    if v > 0
      # Format in scientific notation
      exp = floor(Int, log10(v))
      mantissa = v / 10.0^exp
      
      # Round mantissa to 1 decimal place
      mantissa_rounded = round(mantissa, digits=1)
      
      # Format as LaTeX: mantissa \times 10^{exp}
      if exp == 0
        # For values around 1, just show the mantissa
        push!(yticks_labels, latexstring("$(@sprintf("%.1f", mantissa_rounded))"))
      else
        # Use LaTeX scientific notation without "e" and without "+00"
        push!(yticks_labels, latexstring("$(@sprintf("%.1f", mantissa_rounded))\\cdot 10^{$exp}"))
      end
    else
      push!(yticks_labels, latexstring("0"))
    end
  end
  
  # Define consistent color palette
  order_vals = sort(unique(df.order))
  
  # Generate x-axis ticks
  beta_dim_vals = sort(unique(df.beta_dim))
  xticks_pos = Float64.(beta_dim_vals)
  xticks_labels = [latexstring("$val") for val in beta_dim_vals]
  
  # Set x-axis limits to start from 2
  xlims_left = 1.5  # Start slightly before 2 for better visualization, but data starts at 2
  xlims_right = maximum(beta_dim_vals) + 0.5
  
  plt = plot(xlabel=latexstring("\\beta_{\\mathrm{dim}}"), 
             ylabel=latexstring("\\mathrm{Runtime~(seconds)}"),
             title=latexstring("\\mathrm{(b)~Perturbative~Sketching:~Runtime~vs.~}\\beta_{\\mathrm{dim}}"),
             legend=:topleft,
             xticks=(xticks_pos, xticks_labels),
             xlims=(xlims_left, xlims_right),
             yticks=(yticks_pos, yticks_labels),
             ylims=(ylims_bottom, ylims_top),
             fontsize=fontsize, tickfontsize=fontsize,
             guidefontsize=fontsize, legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm,
             top_margin=14Plots.mm,
             linewidth=2.5, markersize=7, markerstrokewidth=0.1)
  
  # Plot runtime vs beta_dim for each order
  for (idx, order_val) in enumerate(order_vals)
    vline!(plt, [order_val], linestyle=:dash, alpha=0.3, color=:gray, label="")
    sub = df[df.order .== order_val, :]
    idx_sorted = sortperm(sub.beta_dim)
    x_vals = sub.beta_dim[idx_sorted]
    y_vals = sub.runtime_mean[idx_sorted]
    y_err = sub.runtime_std[idx_sorted]
    
    plot!(plt, x_vals, y_vals, yerror=y_err,
          marker=:o, label=latexstring("n_{\\mathrm{Cond.}} = $(order_val)"), 
          color=idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1)
  end
  
  return plt
end

"""
Create combined plot with error and runtime side by side.
"""
function plot_combined(plt_error, plt_runtime; fontsize=18)
  if plt_runtime === nothing
    return plt_error
  end
  
  combined = plot(plt_error, plt_runtime, layout=(1, 2), 
                  size=(1600, 750), dpi=300,
                  plot_title="",  
                  left_margin=8Plots.mm, right_margin=8Plots.mm,
                  bottom_margin=12Plots.mm, top_margin=6Plots.mm)
  
  return combined
end

"""
Transform values for scale break: compress high values, show low values in detail.
"""
function scale_break_transform(y, max_y, threshold=0.1, high_compression=0.1)
  if y < threshold
    return log10(y)
  elseif max_y <= threshold
    return log10(y)
  else
    normalized = (y - threshold) / (max_y - threshold)
    return log10(threshold) + high_compression * normalized
  end
end

"""
Generate y-axis ticks for scale break plot with plain string formatting (no LaTeX).
Returns ticks formatted as powers of 10, showing every second tick.
"""
function generate_yticks(transformed_errors, errors, threshold=0.1, high_compression=0.1; use_latex=false)
  min_transformed = minimum(transformed_errors)
  min_error = minimum(errors)
  
  # Generate power-of-10 ticks for the low range
  min_exp = floor(Int, log10(min_error))
  threshold_exp = floor(Int, log10(threshold))
  
  yticks_pos = Float64[]
  yticks_labels = Any[]
  
  # Add ticks for powers of 10 in the low range, showing every second one
  tick_count = 0
  for exp in min_exp:threshold_exp
    tick_value = 10.0^exp
    if tick_value >= min_error && tick_value < threshold
      tick_pos = log10(tick_value)
      if tick_pos >= min_transformed && tick_pos < log10(threshold)
        if tick_count % 2 == 0  # Show every second tick
          push!(yticks_pos, tick_pos)
          if use_latex
            push!(yticks_labels, latexstring("10^{$exp}"))
          else
            push!(yticks_labels, "10^$exp")
          end
        end
        tick_count += 1
      end
    end
  end
  
  # Add high-value tick if any values exceed threshold
  if any(e >= threshold for e in errors)
    high_tick_position = log10(threshold) + 0.001 * high_compression
    push!(yticks_pos, high_tick_position)
    if use_latex
      push!(yticks_labels, latexstring(">10^{$threshold_exp}"))
    else
      push!(yticks_labels, ">10^$threshold_exp")
    end
  end
  
  # Sort and remove duplicates
  sort_idx = sortperm(yticks_pos)
  yticks_pos = yticks_pos[sort_idx]
  yticks_labels = yticks_labels[sort_idx]
  
  # Remove duplicates within tolerance
  unique_positions = Float64[]
  unique_labels = Any[]
  for (i, pos) in enumerate(yticks_pos)
    if isempty(unique_positions) || abs(pos - unique_positions[end]) > 1e-6
      push!(unique_positions, pos)
      push!(unique_labels, yticks_labels[i])
    end
  end
  
  return unique_positions, unique_labels
end

"""
Sweep a grid over `order` and `beta_dim` and store the max-abs-diff errors as CSV,
then plot error vs `beta_dim` for each `order`.
"""
function run_grid(; n_vertices::Int = 6,
                   β::Real = 1.0,
                   max_order::Union{Int,Nothing} = nothing,
                   sketching_type = Sketching.Perturbative,
                   n_runs::Int = 10,
                   base_seed::Int = 1234,
                   beta_dims::Vector{Int} = [2, 4, 6, 8, 10])
  sketching_label = sketching_type === Sketching.Markov ? "markov" : "perturbative"
  results_path = joinpath(@__DIR__, "beta_choice_results_$(sketching_label).csv")

  # Check if CSV exists and load it, otherwise recompute
  if isfile(results_path)
    println("Loading existing results from: $results_path")
    df = CSV.read(results_path, DataFrame)
    
    # Filter to match current parameters
    df = df[df.n_vertices .== n_vertices .&& df.beta .== β, :]
    
    # Check if we have required columns
    if !(:mean_rel_error in propertynames(df))
      if :max_rel_error in propertynames(df)
        println("Converting old CSV format (max_rel_error) to new format (mean_rel_error)...")
        df[!, :mean_rel_error] = df.max_rel_error
        df[!, :mean_rel_error_std] = zeros(nrow(df))
        CSV.write(results_path, df)
        println("Converted and saved.")
      else
        error("CSV file missing both 'mean_rel_error' and 'max_rel_error' columns.")
      end
    elseif !(:mean_rel_error_std in propertynames(df))
      println("CSV file missing 'mean_rel_error_std' column. Setting to zero.")
      df[!, :mean_rel_error_std] = zeros(nrow(df))
      CSV.write(results_path, df)
    else
      # Replace NaN values in mean_rel_error_std with 0
      df.mean_rel_error_std = [ismissing(x) || (typeof(x) <: Number && isnan(x)) ? 0.0 : x for x in df.mean_rel_error_std]
    end
    
    # Check if DataFrame is empty after filtering
    if nrow(df) == 0
      println("No data matches filter criteria. Computing new results...")
      df = DataFrame()  # Will trigger recomputation
    else
      println("Loaded $(nrow(df)) rows from CSV.")
    end
  else
    println("CSV file not found. Computing new results...")
    df = DataFrame()
  end

  # Recompute if CSV doesn't exist or is empty
  if nrow(df) == 0
    println("Computing results...")
    
    max_o = isnothing(max_order) ? (n_vertices - 1) : min(max_order, n_vertices - 1)
    rows = Vector{NamedTuple}()

    # Run benchmarks using BenchmarkTools for accurate timing
    for order in 1:max_o
      for beta_dim in beta_dims
        sum_ground_truth_vals = Float64[]
        sum_recov_vals = Float64[]
        max_rel_error_vals = Float64[]
        runtimes = Float64[]
        
        mean_rel_error_vals = Float64[]
        std_rel_error_vals = Float64[]
        
        for run_idx in 1:n_runs
          seed = base_seed + run_idx - 1
          
          # Use benchmark_single which handles all setup correctly
          sum_ground_truth, sum_recov, mean_rel_error, std_rel_error, _, _, _ =
            benchmark_single(order, beta_dim; n_vertices=n_vertices, β=β,
                             seed=seed, track_runtime=false)
          
          # Measure runtime separately with BenchmarkTools
          # Reset RNG to ensure consistent initialization
          TTNSsketch.Sketching.Perturbative.RNG[] = nothing
          
          ttns_for_timing = ExampleTopologies.Linear(n_vertices)
          d = length(ttns_for_timing.x_indices)
          set_warn_order(2*d+1)
          p_ground_truth_timing = TTNSsketch.higher_order_probability_dict(ttns_for_timing; β=β, order=order)
          sketching_kwargs_timing = create_sketching_kwargs(beta_dim, seed)
          ttns_recov_timing = deepcopy(ttns_for_timing)
          empty!(ttns_recov_timing.s)
          
          # Use BenchmarkTools to exclude compilation time
          # BenchmarkTools automatically handles warm-up runs, then measures actual runtime
          bench_result = @benchmark CoreDeterminingEquations.compute_Gks!(
            $p_ground_truth_timing, $ttns_recov_timing; 
            sketching_kwargs=$sketching_kwargs_timing
          ) evals=1 seconds=0.1  # seconds parameter ensures enough samples after warm-up
          
          runtime_seconds = mean(bench_result.times) / 1e9  # Convert nanoseconds to seconds
          
          push!(sum_ground_truth_vals, sum_ground_truth)
          push!(sum_recov_vals, sum_recov)
          push!(mean_rel_error_vals, mean_rel_error)
          push!(std_rel_error_vals, std_rel_error)
          push!(runtimes, runtime_seconds)
        end
        
        push!(rows, (
          n_vertices = n_vertices,
          beta = β,
          order = order,
          beta_dim = beta_dim,
          sum_ground_truth = mean(sum_ground_truth_vals),
          sum_recov = mean(sum_recov_vals),
          mean_rel_error = mean(mean_rel_error_vals),  # Mean across runs (should be similar)
          mean_rel_error_std = mean(std_rel_error_vals),  # Mean of std across points (variance of pointwise errors) - this is what we want for error bars!
          runtime_mean = mean(runtimes),
          runtime_std = std(runtimes),
          runtime_min = minimum(runtimes),
          runtime_max = maximum(runtimes),
        ))
      end
    end

    df = DataFrame(rows)
    
    CSV.write(results_path, df)
    println("Results saved to: $results_path")
  end

  # Create combined plot
  plt_error = create_error_plot(df, sketching_label; fontsize=18)
  plt_runtime = create_runtime_plot(df, sketching_label; n_vertices=n_vertices, β=β, fontsize=18)
  if plt_runtime !== nothing
    plt_combined = plot_combined(plt_error, plt_runtime; fontsize=18)
    output_pdf_combined = joinpath(@__DIR__, "beta_choice_combined_$(sketching_label).pdf")
    savefig(plt_combined, output_pdf_combined)
    println("Combined plot saved to: $output_pdf_combined")
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  println("="^60)
  println("Running Perturbative sketching beta_dim benchmark...")
  println("="^60)
  run_grid(n_vertices=8, max_order=7, sketching_type=Sketching.Perturbative, 
           n_runs=1, base_seed=42, beta_dims=[2, 4, 6, 8, 10])
  
  println("\n" * "="^60)
  println("Plots generated:")
  println("  - beta_choice_combined_perturbative.pdf")
  println("="^60)
end

