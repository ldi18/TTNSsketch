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

"""
Create sketching kwargs for Markov sketching.
"""
function create_sketching_kwargs(order_recovery::Int, seed::Int)
  return Dict{Symbol, Any}(
    :sketching_type => Sketching.Markov,
    :order => order_recovery,
    :seed => seed
  )
end

"""
Run a single benchmark for given orders and return summary statistics.
Returns `(sum_ground_truth, sum_recov, max_rel_error, max_abs_diff, ttns_recov, p_ground_truth, runtime_seconds, runtime_dict)`.
If `track_runtime=true`, `runtime_dict` contains per-site runtime information.
Uses report_errors to get max_rel_error (maximum relative error over all points above threshold).
"""
function benchmark_single(order::Int, order_recovery::Int;
                          n_vertices::Int = 6,
                          β::Real = 1.0,
                          seed::Int = 1234,
                          track_runtime::Bool = false)
  # Pre-create topology and ground truth outside timing
  ttns = ExampleTopologies.Linear(n_vertices)
  p_ground_truth = TTNSsketch.higher_order_probability_dict(ttns; β=β, order=order)

  sketching_kwargs = create_sketching_kwargs(order_recovery, seed)
  ttns_recov = deepcopy(ttns)

  # Only measure compute_Gks! time, not dictionary creation
  start_time = time()
  result = CoreDeterminingEquations.compute_Gks!(p_ground_truth, ttns_recov; 
                                                 sketching_kwargs=sketching_kwargs,
                                                 track_runtime=track_runtime)
  runtime_seconds = time() - start_time
  runtime_dict = track_runtime ? result : nothing

  keys_all = collect(keys(p_ground_truth))
  sum_ground_truth = sum(values(p_ground_truth))
  
  # Filter keys to only include those with probability above threshold (to avoid unreliable relative errors for very small values)
  # Threshold: 1e-10 (well above machine precision ~1e-15, but filters out truly tiny probabilities)
  prob_threshold = 1e-10
  keys_filtered = [key for key in keys_all if p_ground_truth[key] > prob_threshold]
  
  # Use report_errors to get maximum relative error (over all points above threshold)
  error_stats = report_errors(p_ground_truth, ttns_recov, keys_filtered; print_sections=false)
  max_rel_error = isnan(error_stats.overall.max) ? 0.0 : error_stats.overall.max
  
  # Also compute sum_recov and max_abs_diff for plotting
  sum_recov = 0.0
  max_abs_diff = 0.0
  for key in keys_all
    p_ref = p_ground_truth[key]
    p_recov = TTNSsketch.evaluate(ttns_recov, key)
    sum_recov += p_recov
    max_abs_diff = max(max_abs_diff, abs(p_ref - p_recov))
  end

  return sum_ground_truth, sum_recov, max_rel_error, max_abs_diff, ttns_recov, p_ground_truth, runtime_seconds, runtime_dict
end

function run_benchmark(; n_vertices::Int = 6,
                        β::Real = 1.0,
                        order::Int = 1,
                        order_recovery::Int = 1,
                        seed::Int = 1234)
  sum_ground_truth, sum_recov, max_rel_error, max_abs_diff, ttns_recov, p_ground_truth, runtime_seconds =
    benchmark_single(order, order_recovery; n_vertices=n_vertices, β=β, seed=seed)

  keys_all = collect(keys(p_ground_truth))
  report_errors(p_ground_truth, ttns_recov, keys_all; prob_precision=8, rel_precision=4)

  @printf("Sum of ground truth probabilities: %.12f\n", sum_ground_truth)
  @printf("Sum of recovered TTNS probabilities: %.12f\n", sum_recov)
  @printf("Max relative error: %.6e\n", max_rel_error)
  @printf("Runtime: %.4f seconds\n", runtime_seconds)
  @printf("order = %d, order_recovery = %d\n", order, order_recovery)
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
Create error plot object (for combined plots).
"""
function create_error_plot(df; threshold=0.1, high_compression=0.1, fontsize=18)
  # Use max_rel_error (maximum relative error over all points)
  error_col = :max_rel_error in propertynames(df) ? :max_rel_error : error("DataFrame must contain max_rel_error column")
  max_error = maximum(df[!, error_col])
  transformed_errors = [scale_break_transform(y, max_error, threshold, high_compression) for y in df[!, error_col]]
  df_transformed = copy(df)
  df_transformed[!, :transformed_error] = transformed_errors
  
  # Use LaTeX labels for proper rendering of formulas
  yticks_pos, yticks_labels = generate_yticks(transformed_errors, df[!, error_col], threshold, high_compression; use_latex=true)
  
  # Calculate y-axis limits: extend slightly above max to ensure all dots are visible, but cut off empty space
  min_y = minimum(transformed_errors)
  max_y = maximum(transformed_errors)
  y_range = max_y - min_y
  ylims_bottom = min_y - 0.02 * y_range  # Small padding at bottom
  ylims_top = max_y + 0.05 * y_range  # Small padding at top to ensure all dots visible, but less than default
  
  # Define consistent color palette (same order for both plots)
  order_vals = sort(unique(df.order))
  
  # Generate x-axis ticks with LaTeX strings
  order_recovery_vals = sort(unique(df.order_recovery))
  xticks_pos = Float64.(order_recovery_vals)
  xticks_labels = [latexstring("$val") for val in order_recovery_vals]
  
  # Use Plots.jl default color scheme - colors will be consistent across plots
  # by using the same order index
  plt = plot(xlabel=latexstring("\\mathrm{Markov~Sketching~Order}~n_{\\mathrm{Sketch}}"), 
             ylabel=latexstring("\\mathrm{Max.~rel.~error}"),
             title=latexstring("\\mathrm{(a)~Max.~Rel.~Error~vs.~Markov~Sketching~Order~}n_{\\mathrm{Sketch}}"),
             legend=false,  # No legend in left plot
             xticks=(xticks_pos, xticks_labels),
             yticks=(yticks_pos, yticks_labels),
             ylims=(ylims_bottom, ylims_top),  # Set y-axis limits
             fontsize=fontsize, tickfontsize=fontsize,
             guidefontsize=fontsize, legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm,  # Add padding to prevent x-axis label cutoff
             top_margin=14Plots.mm,  # Add padding to prevent title cutoff
             linewidth=2.5, markersize=7, markerstrokewidth=0.1)
  
  for (idx, order_val) in enumerate(order_vals)
    vline!(plt, [order_val], linestyle=:dash, alpha=0.3, color=:gray, label="")
    sub = df_transformed[df_transformed.order .== order_val, :]
    idx_sorted = sortperm(sub.order_recovery)
    # Use color index to ensure consistent colors across plots
    plot!(plt, sub.order_recovery[idx_sorted], sub.transformed_error[idx_sorted],
          marker=:o, label=latexstring("n_{\\mathrm{Cond.}} = $(order_val)"), 
          color=idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1)
  end
  
  return plt
end

"""
Create max absolute error plot object (for combined plots).
"""
function create_max_abs_error_plot(df; threshold=0.1, high_compression=0.1, fontsize=18)
  # Use max_abs_diff (maximum absolute error over all points)
  error_col = :max_abs_diff in propertynames(df) ? :max_abs_diff : error("DataFrame must contain max_abs_diff column")
  max_error = maximum(df[!, error_col])
  transformed_errors = [scale_break_transform(y, max_error, threshold, high_compression) for y in df[!, error_col]]
  df_transformed = copy(df)
  df_transformed[!, :transformed_error] = transformed_errors
  
  # Use LaTeX labels for proper rendering of formulas
  yticks_pos, yticks_labels = generate_yticks(transformed_errors, df[!, error_col], threshold, high_compression; use_latex=true)
  
  # Calculate y-axis limits
  min_y = minimum(transformed_errors)
  max_y = maximum(transformed_errors)
  y_range = max_y - min_y
  ylims_bottom = min_y - 0.02 * y_range
  ylims_top = max_y + 0.05 * y_range
  
  # Define consistent color palette (same order for both plots)
  order_vals = sort(unique(df.order))
  
  # Generate x-axis ticks with LaTeX strings
  order_recovery_vals = sort(unique(df.order_recovery))
  xticks_pos = Float64.(order_recovery_vals)
  xticks_labels = [latexstring("$val") for val in order_recovery_vals]
  
  plt = plot(xlabel=latexstring("\\mathrm{Markov~Sketching~Order}~n_{\\mathrm{Sketch}}"), 
             ylabel=latexstring("\\mathrm{Max.~abs.~error}"),
             title=latexstring("\\mathrm{(b)~Max.~Abs.~Error~vs.~Markov~Sketching~Order~}n_{\\mathrm{Sketch}}"),
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
    vline!(plt, [order_val], linestyle=:dash, alpha=0.3, color=:gray, label="")
    sub = df_transformed[df_transformed.order .== order_val, :]
    idx_sorted = sortperm(sub.order_recovery)
    plot!(plt, sub.order_recovery[idx_sorted], sub.transformed_error[idx_sorted],
          marker=:o, label=latexstring("n_{\\mathrm{Cond.}} = $(order_val)"), 
          color=idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1)
  end
  
  return plt
end

"""
Create max/min probability plot object (for combined plots).
"""
function create_prob_range_plot(df; fontsize=18)
  # Define consistent color palette (same order for both plots)
  order_vals = sort(unique(df.order))
  
  # Generate x-axis ticks with LaTeX strings - use model order (n_Cond) instead of sketching order
  xticks_pos = Float64.(order_vals)
  xticks_labels = [latexstring("$val") for val in order_vals]
  
  # Get y-axis range
  all_max_probs = df.max_prob[.!ismissing.(df.max_prob)]
  all_min_probs = df.min_prob[.!ismissing.(df.min_prob)]
  min_y = minimum(vcat(all_max_probs, all_min_probs))
  max_y = maximum(vcat(all_max_probs, all_min_probs))
  ylims_bottom = 1e-14
  ylims_top = 1
  
  # Generate y-axis ticks for log scale
  data_min_exp = isfinite(log10(min_y)) ? floor(Int, log10(min_y)) : -3
  data_max_exp = isfinite(log10(max_y)) ? ceil(Int, log10(max_y)) : 1
  yticks_pos_all = [10.0^exp for exp in (data_min_exp - 1):(data_max_exp + 1)]
  yticks_labels_all = [latexstring("10^{$exp}") for exp in (data_min_exp - 1):(data_max_exp + 1)]
  
  # Filter ticks to be within ylims
  valid_indices = [i for i in 1:length(yticks_pos_all) if yticks_pos_all[i] >= ylims_bottom && yticks_pos_all[i] <= ylims_top]
  if isempty(valid_indices)
    # If no ticks in range, use all ticks
    valid_indices = 1:length(yticks_pos_all)
  end
  yticks_pos = yticks_pos_all[valid_indices]
  yticks_labels = yticks_labels_all[valid_indices]
  
  plt = plot(xlabel=latexstring("\\mathrm{Model~Order}~n_{\\mathrm{Cond}}"), 
             ylabel=latexstring("f(x)~\\mathrm{Value~range}"),
             title=latexstring("\\mathrm{(c)~Function~value~range~vs.~Model~Order~}n_{\\mathrm{Cond.}} = n_{\\mathrm{Sketch}}"),
             legend=:bottomleft,
             xticks=(xticks_pos, xticks_labels),
             yscale=:log10,
             yticks=(yticks_pos, yticks_labels),
             ylims=(ylims_bottom, ylims_top),
             fontsize=fontsize, tickfontsize=fontsize,
             guidefontsize=fontsize, legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm,
             top_margin=14Plots.mm,
             linewidth=2.5, markersize=7, markerstrokewidth=0.1)
  
  # Plot max and min probabilities for each order
  # Since max_prob and min_prob are constant for a given order (regardless of order_recovery),
  # we can use any row for each order
  # Collect all values first to plot as connected lines
  x_vals = Float64[]
  max_probs = Float64[]
  min_probs = Float64[]
  
  for (idx, order_val) in enumerate(order_vals)
    sub = df[df.order .== order_val, :]
    if !isempty(sub)
      # Use the first row's values (they're all the same for a given order)
      first_row = sub[1, :]
      max_prob = first_row.max_prob
      min_prob = first_row.min_prob
      
      if !ismissing(max_prob) && !ismissing(min_prob) && !isnan(max_prob) && !isnan(min_prob)
        push!(x_vals, Float64(order_val))
        push!(max_probs, max_prob)
        push!(min_probs, min_prob)
      end
    end
  end
  
  # Plot as connected lines
  if !isempty(x_vals)
    plot!(plt, x_vals, max_probs,
          marker=:o, linestyle=:solid, label=latexstring("\\max"), 
          color=1, linewidth=2.5, markersize=7, markerstrokewidth=0.1)
    plot!(plt, x_vals, min_probs,
          marker=:ut, linestyle=:dash, label=latexstring("\\min"), 
          color=1, linewidth=2.5, markersize=7, markerstrokewidth=0.1, alpha=0.7)
  end
  
  return plt
end

"""
Compute SVD scaling function: order_scaling(n) = n^2 * (n+1)
This represents the approximate scaling of SVD operations.
"""
function order_scaling(n)
    return n^2 * (n + 1)
end

"""
Create runtime plot object (for combined plots).
Shows runtime vs order_recovery as a line plot with connected dots.
"""
function create_runtime_plot(df; n_vertices::Int, β::Real, fontsize=18)
  # Check if runtime data exists
  if !(:runtime_mean in propertynames(df)) || all(ismissing.(df.runtime_mean))
    println("WARNING: No runtime data available - skipping runtime plot")
    return nothing
  end
  
  # Get y-axis range to generate ticks
  all_runtimes = df.runtime_mean[.!ismissing.(df.runtime_mean)]
  if isempty(all_runtimes)
    println("WARNING: No valid runtime data - skipping runtime plot")
    return nothing
  end
  
  min_runtime = minimum(all_runtimes)
  max_runtime = maximum(all_runtimes)
  
  # Account for error bars when calculating max
  max_runtime_with_error = max_runtime
  for (i, r) in enumerate(df.runtime_mean)
    if !ismissing(r) && !ismissing(df.runtime_std[i])
      max_runtime_with_error = max(max_runtime_with_error, r + df.runtime_std[i])
    end
  end
  
  # Calculate y-axis limits for log scale: extend slightly above max to ensure all dots and error bars are visible
  # For log scale, we multiply by a factor instead of adding
  ylims_bottom = max(min_runtime * 0.8, 1e-20)  # Small padding at bottom, ensure positive
  ylims_top = max_runtime_with_error * 1.3  # Extend above max to ensure all dots visible, but cut off empty space
  
  # Generate ticks based on the data range - simpler and more reliable
  # Calculate exponent range from actual data
  data_min_exp = isfinite(log10(min_runtime)) ? floor(Int, log10(min_runtime)) : -3
  data_max_exp = isfinite(log10(max_runtime_with_error)) ? ceil(Int, log10(max_runtime_with_error)) : 1
  
  # Generate integer-order ticks from one order below to one order above the data range
  yticks_pos_all = [10.0^exp for exp in (data_min_exp - 1):(data_max_exp + 1)]
  yticks_labels_all = [latexstring("10^{$exp}") for exp in (data_min_exp - 1):(data_max_exp + 1)]
  
  # Filter ticks to only show 10^-2 to 10^0
  valid_indices = [i for i in 1:length(yticks_pos_all) if yticks_pos_all[i] >= 1e-2 && yticks_pos_all[i] <= 1e0]
  if isempty(valid_indices)
    # If no ticks in range, use default range
    yticks_pos = yticks_pos_all
    yticks_labels = yticks_labels_all
  else
    yticks_pos = yticks_pos_all[valid_indices]
    yticks_labels = yticks_labels_all[valid_indices]
  end
  
  # Define consistent color palette (same order as error plot)
  order_vals = sort(unique(df.order))
  
  # Generate x-axis ticks with LaTeX strings (same as error plot)
  order_recovery_vals = sort(unique(df.order_recovery))
  xticks_pos = Float64.(order_recovery_vals)
  xticks_labels = [latexstring("$val") for val in order_recovery_vals]
  
  plt = plot(xlabel=latexstring("\\mathrm{Markov~Sketching~Order}~n_{\\mathrm{Sketch}}"), 
             ylabel=latexstring("\\mathrm{Runtime~(seconds)}"),
             title=latexstring("\\mathrm{(d)~Runtime~vs.~Markov~Sketching~Order~}n_{\\mathrm{Sketch}}"),
             legend=:bottomright,
             xticks=(xticks_pos, xticks_labels),
             yscale=:log10, yticks=(yticks_pos, yticks_labels),
             ylims=(ylims_bottom, ylims_top),  # Set y-axis limits
             fontsize=fontsize, tickfontsize=fontsize,
             guidefontsize=fontsize, legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm,  # Add padding to prevent x-axis label cutoff
             top_margin=14Plots.mm,  # Add padding to prevent title cutoff
             linewidth=2.5, markersize=7, markerstrokewidth=0.1)
  
  # Plot runtime vs order_recovery for each order
  for (idx, order_val) in enumerate(order_vals)
    vline!(plt, [order_val], linestyle=:dash, alpha=0.3, color=:gray, label="")
    sub = df[df.order .== order_val, :]
    idx_sorted = sortperm(sub.order_recovery)
    x_vals = sub.order_recovery[idx_sorted]
    y_vals = sub.runtime_mean[idx_sorted]
    y_err = sub.runtime_std[idx_sorted]
    
    # Use same color index as error plot to ensure consistent colors
      plot!(plt, x_vals, y_vals, yerror=y_err,
            marker=:o, label=latexstring("n_{\\mathrm{Cond.}} = $(order_val)"), 
            color=idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1)
  end
  
  # Add SVD scaling line: compute scaling values and scale so value at n_Sketch=8 is 3
  if !isempty(order_recovery_vals)
    # Compute scaling values for all order_recovery values
    scaling_values = [order_scaling(n) for n in order_recovery_vals]
    
    # Find scaling factor: scale so that value at order_recovery=8 is 2
    target_order = 8
    target_value = 2.0
    
    # Compute scaling factor based on target_order (even if it's not in the data)
    target_scaling = order_scaling(target_order)
    if target_scaling > 0
      scaling_factor = target_value / target_scaling
      
      # Compute scaled values
      scaled_values = [s * scaling_factor for s in scaling_values]
      
      # Plot as dotted line
      plot!(plt, order_recovery_vals, scaled_values,
            linestyle=:dot, linewidth=2.5, 
            label=latexstring("\\mathcal{O}(\\mathrm{SVD})"),
            color=:black, alpha=0.7)
    end
  end
  
  return plt
end

"""
Create combined plot with 2x2 layout: max rel error, max abs error, runtime, prob range.
"""
function plot_combined(plt_error, plt_max_abs_error, plt_runtime, plt_prob_range; fontsize=18)
  if plt_runtime === nothing
    # Fallback to 1x2 if runtime not available
    return plot(plt_error, plt_max_abs_error, layout=(1, 2), 
                size=(1600, 750), dpi=300,
                plot_title="",  
                left_margin=8Plots.mm, right_margin=8Plots.mm,
                bottom_margin=12Plots.mm, top_margin=6Plots.mm)
  end
  
  combined = plot(plt_error, plt_max_abs_error, plt_prob_range, plt_runtime, 
                  layout=(2, 2), 
                  size=(1600, 1500), dpi=300,
                  plot_title="",  
                  left_margin=8Plots.mm, right_margin=8Plots.mm,
                  bottom_margin=12Plots.mm, top_margin=6Plots.mm)
  
  return combined
end

"""
Sweep a grid over `order` and `order_recovery` and store the max-abs-diff errors as CSV,
then plot error vs `order_recovery` for each `order`.
"""
function run_grid(; n_vertices::Int = 6,
                   β::Real = 1.0,
                   max_order::Union{Int,Nothing} = nothing,
                   n_runs::Int = 10,
                   base_seed::Int = 1234)
  sketching_label = "markov"
  results_path = joinpath(@__DIR__, "higher_order_markov_results_$(sketching_label).csv")

  # Auto-reload if CSV exists (always use existing data if available)
  csv_exists = isfile(results_path)
  if csv_exists
    println("Loading existing results from: $results_path")
    df = CSV.read(results_path, DataFrame)
  else
    # Compute if CSV doesn't exist
    
    max_o = isnothing(max_order) ? (n_vertices - 1) : min(max_order, n_vertices - 1)
    rows = Vector{NamedTuple}()

    for order in 1:max_o
      for order_recovery in 1:(n_vertices - 1)
        sum_ground_truth_vals = Float64[]
        sum_recov_vals = Float64[]
        max_rel_error_vals = Float64[]
        max_abs_diff_vals = Float64[]
        max_prob_vals = Float64[]
        min_prob_vals = Float64[]
        runtimes = Float64[]
        
        for run_idx in 1:n_runs
          seed = base_seed + run_idx - 1
          
          # Pre-create topology and ground truth outside timing
          ttns = ExampleTopologies.Linear(n_vertices)
          p_ground_truth = TTNSsketch.higher_order_probability_dict(ttns; β=β, order=order)
          sketching_kwargs = create_sketching_kwargs(order_recovery, seed)
          ttns_recov = deepcopy(ttns)
          
          # Use BenchmarkTools to measure only compute_Gks! time (handles warm-up automatically)
          bench_result = @benchmark CoreDeterminingEquations.compute_Gks!(
            $p_ground_truth, $ttns_recov; 
            sketching_kwargs=$sketching_kwargs
          ) evals=1
          
          # Compute metrics directly from the already-computed ttns_recov (modified in place by compute_Gks!)
          keys_all = collect(keys(p_ground_truth))
          sum_ground_truth = sum(values(p_ground_truth))
          
          # Filter keys to only include those with probability above threshold (to avoid unreliable relative errors for very small values)
          # Threshold: 1e-10 (well above machine precision ~1e-15, but filters out truly tiny probabilities)
          prob_threshold = 1e-10
          keys_filtered = [key for key in keys_all if p_ground_truth[key] > prob_threshold]
          
          # Use report_errors to get maximum relative error (over all points above threshold)
          error_stats = report_errors(p_ground_truth, ttns_recov, keys_filtered; print_sections=false)
          max_rel_error = isnan(error_stats.overall.max) ? 0.0 : error_stats.overall.max
          
          # Also compute sum_recov, max_abs_diff, and max/min probabilities for plotting
          sum_recov = 0.0
          max_abs_diff = 0.0
          for key in keys_all
            p_ref = p_ground_truth[key]
            p_recov = TTNSsketch.evaluate(ttns_recov, key)
            sum_recov += p_recov
            max_abs_diff = max(max_abs_diff, abs(p_ref - p_recov))
          end
          
          max_prob = maximum(values(p_ground_truth))
          min_prob = minimum(values(p_ground_truth))
          
          runtime_seconds = mean(bench_result.times) / 1e9  # Convert nanoseconds to seconds
          
          push!(sum_ground_truth_vals, sum_ground_truth)
          push!(sum_recov_vals, sum_recov)
          push!(max_rel_error_vals, max_rel_error)
          push!(max_abs_diff_vals, max_abs_diff)
          push!(max_prob_vals, max_prob)
          push!(min_prob_vals, min_prob)
          push!(runtimes, runtime_seconds)
        end
        
        push!(rows, (
          n_vertices = n_vertices,
          beta = β,
          order = order,
          order_recovery = order_recovery,
          sum_ground_truth = mean(sum_ground_truth_vals),
          sum_recov = mean(sum_recov_vals),
          max_rel_error = mean(max_rel_error_vals),
          max_abs_diff = mean(max_abs_diff_vals),
          max_prob = mean(max_prob_vals),
          min_prob = mean(min_prob_vals),
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
  df = df[df.n_vertices .== n_vertices .&& df.beta .== β, :]

  # Create combined plot
  plt_error = create_error_plot(df; fontsize=18)
  plt_max_abs_error = create_max_abs_error_plot(df; fontsize=18)
  plt_runtime = create_runtime_plot(df; n_vertices=n_vertices, β=β, fontsize=18)
  plt_prob_range = create_prob_range_plot(df; fontsize=18)
  if plt_runtime !== nothing
    plt_combined = plot_combined(plt_error, plt_max_abs_error, plt_runtime, plt_prob_range; fontsize=18)
    output_pdf_combined = joinpath(@__DIR__, "higher_order_markov_combined_$(sketching_label).pdf")
    savefig(plt_combined, output_pdf_combined)
    println("Combined plot saved to: $output_pdf_combined")
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  println("="^60)
  println("Running Markov sketching benchmark...")
  println("="^60)
  run_grid(n_vertices=9, max_order=8, n_runs=1, base_seed=123)   # n_vertices=9, max_order=7
  
  println("\n" * "="^60)
  println("Plots generated:")
  println("  - higher_order_markov_combined_markov.pdf")
  println("="^60)
end

