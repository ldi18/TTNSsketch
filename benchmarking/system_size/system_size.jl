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
Create sketching kwargs for Markov sketching.
"""
function create_sketching_kwargs(order::Int, seed::Int; enforce_non_recursive::Bool = false)
  kwargs = Dict{Symbol, Any}(
    :sketching_type => Sketching.Markov,
    :order => order,
    :seed => seed
  )
  if enforce_non_recursive
    kwargs[:enforce_non_recursive] = true
  end
  return kwargs
end

"""
Run a single benchmark for given sketching order and system size.
Returns high-level error metrics and runtime.
"""
function benchmark_single(sketching_order::Int, n_vertices::Int;
                          β::Real = 1.0,
                          seed::Int = 1234,
                          enforce_non_recursive::Bool = false)
  ttns = ExampleTopologies.Linear(n_vertices)
  # Use the same order for ground truth correlations as the sketching order
  p_ground_truth = TTNSsketch.higher_order_probability_dict(ttns; β=β, order=sketching_order)
  
  sketching_kwargs = create_sketching_kwargs(sketching_order, seed; enforce_non_recursive=enforce_non_recursive)
  ttns_recov = deepcopy(ttns)
  
  start_time = time()
  CoreDeterminingEquations.compute_Gks!(p_ground_truth, ttns_recov; sketching_kwargs=sketching_kwargs)
  runtime_seconds = time() - start_time
  
  # Use report_errors to get maximum relative error (over all points above threshold)
  # Filter keys to only include those with probability above threshold (to avoid unreliable relative errors for very small values)
  # Threshold: 1e-10 (well above machine precision ~1e-15, but filters out truly tiny probabilities)
  keys_all = collect(keys(p_ground_truth))
  prob_threshold = 1e-10
  keys_filtered = [key for key in keys_all if p_ground_truth[key] > prob_threshold]
  error_stats = report_errors(p_ground_truth, ttns_recov, keys_filtered; print_sections=false)
  max_rel_error = isnan(error_stats.overall.max) ? 0.0 : error_stats.overall.max
  
  return max_rel_error, runtime_seconds
end

"""
Sweep over system sizes and sketching orders, then plot high-level error vs system size.
"""
function run_benchmark(; max_n::Int = 12,
                       β::Real = 1.0,
                       sketching_orders::Vector{Int} = [1, 2, 3],
                       n_runs::Int = 5,
                       base_seed::Int = 1234)
  results_path = joinpath(@__DIR__, "system_size_results.csv")
  set_warn_order(max_n+1)
  
  # Auto-reload if CSV exists (always use existing data if available)
  csv_exists = isfile(results_path)
  if csv_exists
    println("Loading existing results from: $results_path")
    df = CSV.read(results_path, DataFrame)
    # Handle backward compatibility: if is_recursive column doesn't exist, add it (default to false)
    if !(:is_recursive in propertynames(df))
      df.is_recursive = false
    end
  else
    # Compute if CSV doesn't exist
    rows = Vector{NamedTuple}()
    
    # Run benchmarks using BenchmarkTools for accurate timing (handles warm-up automatically)
    for sketching_order in sketching_orders
      # For order 1, run both recursive and non-recursive cases
      recursive_cases = sketching_order == 1 ? [false, true] : [false]
      
      for is_recursive in recursive_cases
        for n in 3:max_n
          max_rel_errors = Float64[]
          max_abs_diffs = Float64[]
          max_probs = Float64[]
          min_probs = Float64[]
          runtimes = Float64[]
          
          for run_idx in 1:n_runs
            seed = base_seed + run_idx - 1
            
            # Pre-create topology and ground truth outside timing
            ttns = ExampleTopologies.Linear(n)
            p_ground_truth = TTNSsketch.higher_order_probability_dict(ttns; β=β, order=sketching_order)
            enforce_non_recursive = !is_recursive  # If recursive, don't enforce non-recursive
            sketching_kwargs = create_sketching_kwargs(sketching_order, seed; enforce_non_recursive=enforce_non_recursive)
            ttns_recov = deepcopy(ttns)
            
            # Use BenchmarkTools to measure only compute_Gks! time (handles warm-up automatically)
            bench_result = @benchmark CoreDeterminingEquations.compute_Gks!(
              $p_ground_truth, $ttns_recov; 
              sketching_kwargs=$sketching_kwargs
            ) evals=1
            
            # Compute metrics directly from the already-computed ttns_recov (modified in place by compute_Gks!)
            keys_all = collect(keys(p_ground_truth))
            
            # Filter keys to only include those with probability above threshold (to avoid unreliable relative errors for very small values)
            # Threshold: 1e-10 (well above machine precision ~1e-15, but filters out truly tiny probabilities)
            prob_threshold = 1e-10
            keys_filtered = [key for key in keys_all if p_ground_truth[key] > prob_threshold]
            
            error_stats = report_errors(p_ground_truth, ttns_recov, keys_filtered; print_sections=false)
            max_rel_error = isnan(error_stats.overall.max) ? 0.0 : error_stats.overall.max
            
            # Also compute max_abs_diff and max/min probabilities for plotting
            max_abs_diff = 0.0
            for key in keys_all
              p_ref = p_ground_truth[key]
              p_recov = TTNSsketch.evaluate(ttns_recov, key)
              max_abs_diff = max(max_abs_diff, abs(p_ref - p_recov))
            end
            
            max_prob = maximum(values(p_ground_truth))
            min_prob = minimum(values(p_ground_truth))
            
            runtime_seconds = mean(bench_result.times) / 1e9  # Convert nanoseconds to seconds
            
            push!(max_rel_errors, max_rel_error)
            push!(max_abs_diffs, max_abs_diff)
            push!(max_probs, max_prob)
            push!(min_probs, min_prob)
            push!(runtimes, runtime_seconds)
          end
          
          push!(rows, (
            sketching_order = sketching_order,
            is_recursive = is_recursive,
            n_vertices = n,
            beta = β,
            max_rel_error_mean = mean(max_rel_errors),
            max_rel_error_std = std(max_rel_errors),
            max_abs_diff_mean = mean(max_abs_diffs),
            max_abs_diff_std = std(max_abs_diffs),
            max_prob = mean(max_probs),
            min_prob = mean(min_probs),
            runtime_mean = mean(runtimes),
            runtime_std = std(runtimes),
            runtime_min = minimum(runtimes),
            runtime_max = maximum(runtimes),
          ))
        end
      end
    end
    
    df = DataFrame(rows)
    CSV.write(results_path, df)
    println("Results saved to: $results_path")
  end
  
  # Filter to desired β
  df = df[df.beta .== β, :]
  
  # Create plots
  plt_error = create_error_plot(df, sketching_orders; fontsize=18)
  plt_max_abs_error = create_max_abs_error_plot(df, sketching_orders; fontsize=18)
  plt_runtime = create_runtime_plot(df, sketching_orders; fontsize=18)
  plt_prob_range = create_prob_range_plot(df, sketching_orders; fontsize=18)
  
  # Create combined plot
  plt_combined = plot_combined(plt_error, plt_max_abs_error, plt_runtime, plt_prob_range; fontsize=18)
  
  # Save individual plots
  output_pdf_error = joinpath(@__DIR__, "system_size_error_vs_n.pdf")
  savefig(plt_error, output_pdf_error)
  println("Error plot saved to: $output_pdf_error")
  
  output_pdf_runtime = joinpath(@__DIR__, "system_size_runtime_vs_n.pdf")
  if plt_runtime !== nothing
    savefig(plt_runtime, output_pdf_runtime)
    println("Runtime plot saved to: $output_pdf_runtime")
  end
  
  output_pdf_combined = joinpath(@__DIR__, "system_size_combined.pdf")
  savefig(plt_combined, output_pdf_combined)
  println("Combined plot saved to: $output_pdf_combined")
  
  return df, plt_combined
end

"""
Create error plot showing high-level error vs system size.
"""
function create_error_plot(df, sketching_orders; fontsize=18)
  # Get y-axis range to generate ticks (using max errors)
  all_errors = vcat([df[df.sketching_order .== order, :].max_rel_error_mean for order in sketching_orders]...)
  min_error = minimum(all_errors)
  max_error = maximum(all_errors)
  min_error = max(min_error, 1e-20)
  
  # Calculate y-axis limits
  ylims_bottom = min_error * 0.8
  ylims_top = max_error * 1.3
  
  # Generate ticks based on data range
  data_min_exp = isfinite(log10(min_error)) ? floor(Int, log10(min_error)) : -3
  data_max_exp = isfinite(log10(max_error)) ? ceil(Int, log10(max_error)) : 1
  yticks_pos_all = [10.0^exp for exp in (data_min_exp - 1):(data_max_exp + 1)]
  yticks_labels_all = [latexstring("10^{$exp}") for exp in (data_min_exp - 1):(data_max_exp + 1)]
  
  # Filter ticks: remove <= 10^-15 and >= 10^-4
  valid_indices = [i for i in 1:length(yticks_pos_all) if yticks_pos_all[i] > 1e-15 && yticks_pos_all[i] < 1e-4]
  yticks_pos = yticks_pos_all[valid_indices]
  yticks_labels = yticks_labels_all[valid_indices]
  
  # Generate x-axis ticks with LaTeX strings
  n_vals = sort(unique(df.n_vertices))
  xticks_pos = Float64.(n_vals)
  xticks_labels = [latexstring("$val") for val in n_vals]
  
  plt = plot(xlabel=L"\mathrm{System~size~}d", 
             ylabel=latexstring("\\mathrm{Max.~rel.~error}"),
             title=latexstring("\\mathrm{(a)~Max.~Rel.~Error~vs.~System~size~}d"),
             legend=false,
             xticks=(xticks_pos, xticks_labels),
             yscale=:log10, yticks=(yticks_pos, yticks_labels),
             ylims=(ylims_bottom, ylims_top),
             fontsize=fontsize, tickfontsize=fontsize, 
             guidefontsize=fontsize, legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm, top_margin=14Plots.mm,
             linewidth=2.5, markersize=7, markerstrokewidth=0.1)
  
  color_idx = 1
  for order in sketching_orders
    # For order 1, handle both recursive and non-recursive cases
    if order == 1
      # Check if is_recursive column exists
      has_recursive_col = :is_recursive in propertynames(df)
      
      # Recursive case (plot first to appear first in legend)
      if has_recursive_col
        sub_recursive = df[(df.sketching_order .== order) .& (df.is_recursive .== true), :]
        if !isempty(sub_recursive)
          idx_sorted = sortperm(sub_recursive.n_vertices)
          x_vals = sub_recursive.n_vertices[idx_sorted]
          y_vals = Vector{Float64}(sub_recursive.max_rel_error_mean[idx_sorted])
          
          plot!(plt, x_vals, y_vals,
                marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order) \\, \\mathrm{(recursive)}"), 
                color=color_idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1, linestyle=:dot)
          color_idx += 1
        end
      end
      
      # Non-recursive case (plot second)
      if has_recursive_col
        sub_non_recursive = df[(df.sketching_order .== order) .& (df.is_recursive .== false), :]
      else
        # Backward compatibility: if column doesn't exist, all order 1 data is non-recursive
        sub_non_recursive = df[df.sketching_order .== order, :]
      end
      if !isempty(sub_non_recursive)
        idx_sorted = sortperm(sub_non_recursive.n_vertices)
        x_vals = sub_non_recursive.n_vertices[idx_sorted]
        y_vals = Vector{Float64}(sub_non_recursive.max_rel_error_mean[idx_sorted])
        
        plot!(plt, x_vals, y_vals,
              marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order)"), 
              color=color_idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1, linestyle=:solid)
        color_idx += 1
      end
    else
      # For other orders, use standard behavior (always non-recursive)
      # Filter to non-recursive if column exists, otherwise take all
      if :is_recursive in propertynames(df)
        sub = df[(df.sketching_order .== order) .& (df.is_recursive .== false), :]
      else
        sub = df[df.sketching_order .== order, :]
      end
      if !isempty(sub)
        idx_sorted = sortperm(sub.n_vertices)
        x_vals = sub.n_vertices[idx_sorted]
        y_vals = Vector{Float64}(sub.max_rel_error_mean[idx_sorted])
        
        plot!(plt, x_vals, y_vals,
              marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order)"), 
              color=color_idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1)
        color_idx += 1
      end
    end
  end
  
  return plt
end

"""
Create max absolute error plot showing max abs error vs system size.
"""
function create_max_abs_error_plot(df, sketching_orders; fontsize=18)
  # Get y-axis range to generate ticks
  all_errors = vcat([df[df.sketching_order .== order, :].max_abs_diff_mean for order in sketching_orders]...)
  min_error = minimum(all_errors)
  max_error = maximum(all_errors)
  min_error = max(min_error, 1e-20)
  
  # Calculate y-axis limits for log scale
  ylims_bottom = min_error / 100
  ylims_top = max_error * 100
  
  # Generate ticks based on data range - only integer orders
  data_min_exp = isfinite(log10(min_error)) ? floor(Int, log10(min_error)) : -18
  data_max_exp = isfinite(log10(max_error)) ? ceil(Int, log10(max_error)) : -14
  
  # Generate integer order ticks
  yticks_pos = Float64[]
  yticks_labels = Vector{LaTeXString}()
  
  for exp in (data_min_exp - 1):(data_max_exp + 1)
    tick_val = 10.0^exp
    if tick_val > 1e-20 && tick_val >= ylims_bottom && tick_val <= ylims_top
      push!(yticks_pos, tick_val)
      push!(yticks_labels, latexstring("10^{$exp}"))
    end
  end
  
  # If still no ticks, generate a minimal set
  if isempty(yticks_pos)
    # Generate at least a few ticks within the range
    exp_min = floor(Int, log10(ylims_bottom))
    exp_max = ceil(Int, log10(ylims_top))
    yticks_pos = [10.0^exp for exp in exp_min:exp_max if 10.0^exp >= ylims_bottom && 10.0^exp <= ylims_top]
    yticks_labels = [latexstring("10^{$exp}") for exp in exp_min:exp_max if 10.0^exp >= ylims_bottom && 10.0^exp <= ylims_top]
  end
  
  # Generate x-axis ticks with LaTeX strings
  n_vals = sort(unique(df.n_vertices))
  xticks_pos = Float64.(n_vals)
  xticks_labels = [latexstring("$val") for val in n_vals]
  
  plt = plot(xlabel=L"\mathrm{System~size~}d", 
             ylabel=latexstring("\\mathrm{Max.~abs.~error}"),
             title=latexstring("\\mathrm{(b)~Max.~Abs.~Error~vs.~System~size~}d"),
             legend=false,
             xticks=(xticks_pos, xticks_labels),
             yscale=:log10, yticks=(yticks_pos, yticks_labels),
             ylims=(ylims_bottom, ylims_top),
             fontsize=fontsize, tickfontsize=fontsize,
             guidefontsize=fontsize, legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm,
             top_margin=14Plots.mm,
             linewidth=2.5,
             markersize=7,
             markerstrokewidth=0.1)
  
  color_idx = 1
  for order in sketching_orders
    if order == 1
      # For order 1, plot both recursive and non-recursive
      if :is_recursive in propertynames(df)
        sub_recursive = df[(df.sketching_order .== order) .& (df.is_recursive .== true), :]
        sub_non_recursive = df[(df.sketching_order .== order) .& (df.is_recursive .== false), :]
        
        if !isempty(sub_recursive)
          idx_sorted = sortperm(sub_recursive.n_vertices)
          x_vals = sub_recursive.n_vertices[idx_sorted]
          y_vals = Vector{Float64}(sub_recursive.max_abs_diff_mean[idx_sorted])
          
          plot!(plt, x_vals, y_vals,
                marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order),~\\mathrm{recursive}"), 
                color=color_idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1, linestyle=:dot)
          color_idx += 1
        end
        
        if !isempty(sub_non_recursive)
          idx_sorted = sortperm(sub_non_recursive.n_vertices)
          x_vals = sub_non_recursive.n_vertices[idx_sorted]
          y_vals = Vector{Float64}(sub_non_recursive.max_abs_diff_mean[idx_sorted])
          
          plot!(plt, x_vals, y_vals,
                marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order),~\\mathrm{non-recursive}"), 
                color=color_idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1, linestyle=:solid)
          color_idx += 1
        end
      end
    else
      # For other orders, use standard behavior (always non-recursive)
      if :is_recursive in propertynames(df)
        sub = df[(df.sketching_order .== order) .& (df.is_recursive .== false), :]
      else
        sub = df[df.sketching_order .== order, :]
      end
      if !isempty(sub)
        idx_sorted = sortperm(sub.n_vertices)
        x_vals = sub.n_vertices[idx_sorted]
        y_vals = Vector{Float64}(sub.max_abs_diff_mean[idx_sorted])
        
        plot!(plt, x_vals, y_vals,
              marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order)"), 
              color=color_idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1)
        color_idx += 1
      end
    end
  end
  
  return plt
end

"""
Create probability range plot showing max/min probabilities vs system size.
"""
function create_prob_range_plot(df, sketching_orders; fontsize=18)  
  # Get y-axis range
  all_max_probs = df.max_prob[.!ismissing.(df.max_prob)]
  all_min_probs = df.min_prob[.!ismissing.(df.min_prob)]
  
  if isempty(all_max_probs) || isempty(all_min_probs)
    println("WARNING: No valid max_prob or min_prob data - creating empty plot")
    plt = plot(xlabel=L"\mathrm{System~size~}d", 
               ylabel=latexstring("f(x)~\\mathrm{Value~range}"),
               title=latexstring("\\mathrm{(c)}"),
               legend=:bottomleft,
               fontsize=fontsize)
    return plt
  end
  
  min_y = minimum(vcat(all_max_probs, all_min_probs))
  max_y = maximum(vcat(all_max_probs, all_min_probs))
  ylims_bottom = max(min_y * 0.5, 1e-20)  # Ensure positive for log scale
  ylims_top = max_y * 2.0
  
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
  
  # Generate x-axis ticks with LaTeX strings
  n_vals = sort(unique(df.n_vertices))
  xticks_pos = Float64.(n_vals)
  xticks_labels = [latexstring("$val") for val in n_vals]
  
    plt = plot(xlabel=L"\mathrm{System~size~}d", 
               ylabel=latexstring("f(x)~\\mathrm{Value~range}"),
               title=latexstring("\\mathrm{(c)~Function~value~range~vs.~System~size~}d"),
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
  
  # Since max_prob and min_prob are the same for all orders (they're from the original function),
  # we can just plot them once using any order's data
  # Use the first available order's data
  first_order = sketching_orders[1]
  if :is_recursive in propertynames(df)
    sub = df[(df.sketching_order .== first_order) .& (df.is_recursive .== false), :]
  else
    sub = df[df.sketching_order .== first_order, :]
  end
  
  if !isempty(sub)
    idx_sorted = sortperm(sub.n_vertices)
    x_vals = sub.n_vertices[idx_sorted]
    max_probs = sub.max_prob[idx_sorted]
    min_probs = sub.min_prob[idx_sorted]
    
    # Filter out any NaN or missing values
    valid_indices = [i for i in 1:length(x_vals) if !ismissing(max_probs[i]) && !ismissing(min_probs[i]) && !isnan(max_probs[i]) && !isnan(min_probs[i])]
    if !isempty(valid_indices)
      x_vals = x_vals[valid_indices]
      max_probs = max_probs[valid_indices]
      min_probs = min_probs[valid_indices]
      
      plot!(plt, x_vals, max_probs,
            marker=:o, linestyle=:solid, label=latexstring("\\max_x f(x)"), 
            color=1, linewidth=2.5, markersize=7, markerstrokewidth=0.1)
      plot!(plt, x_vals, min_probs,
            marker=:ut, linestyle=:dash, label=latexstring("\\min_x f(x)"), 
            color=1, linewidth=2.5, markersize=7, markerstrokewidth=0.1, alpha=0.7)
    else
      println("WARNING: No valid data points for probability range plot")
    end
  else
    println("WARNING: No data found for order $first_order in probability range plot")
  end
  
  return plt
end

"""
Create runtime plot showing runtime vs system size.
"""
function create_runtime_plot(df, sketching_orders; fontsize=18)
  # Check if runtime data exists
  if !(:runtime_mean in propertynames(df)) || all(ismissing.(df.runtime_mean))
    println("WARNING: No runtime data available - skipping runtime plot")
    return nothing
  end
  
  # Get y-axis range to generate ticks
  all_runtimes = vcat([df[df.sketching_order .== order, :].runtime_mean for order in sketching_orders]...)
  min_runtime = minimum(all_runtimes)
  max_runtime = maximum(all_runtimes)
  min_runtime = max(min_runtime, 1e-20)
  
  # Account for error bars when calculating max
  max_runtime_with_error = max_runtime
  for (i, r) in enumerate(df.runtime_mean)
    if !ismissing(r) && !ismissing(df.runtime_std[i])
      max_runtime_with_error = max(max_runtime_with_error, r + df.runtime_std[i])
    end
  end
  
  # Calculate y-axis limits
  ylims_bottom = max(min_runtime * 0.8, 1e-20)
  ylims_top = max_runtime_with_error * 1.3
  
  # Generate ticks based on data range
  data_min_exp = isfinite(log10(min_runtime)) ? floor(Int, log10(min_runtime)) : -3
  data_max_exp = isfinite(log10(max_runtime_with_error)) ? ceil(Int, log10(max_runtime_with_error)) : 1
  yticks_pos_all = [10.0^exp for exp in (data_min_exp - 1):(data_max_exp + 1)]
  yticks_labels_all = [latexstring("10^{$exp}") for exp in (data_min_exp - 1):(data_max_exp + 1)]
  
  # Filter ticks: remove >= 10^1 and <= 10^-4
  valid_indices = [i for i in 1:length(yticks_pos_all) if yticks_pos_all[i] < 1e1 && yticks_pos_all[i] > 1e-4]
  yticks_pos = yticks_pos_all[valid_indices]
  yticks_labels = yticks_labels_all[valid_indices]
  
  # Generate x-axis ticks with LaTeX strings
  n_vals = sort(unique(df.n_vertices))
  xticks_pos = Float64.(n_vals)
  xticks_labels = [latexstring("$val") for val in n_vals]
  
  plt = plot(xlabel=L"\mathrm{System~size~}d", 
             ylabel=latexstring("\\mathrm{Runtime~(seconds)}"),
             title=latexstring("\\mathrm{(d)~Runtime~vs.~System~size~}d"),
             legend=:bottomright,
             xticks=(xticks_pos, xticks_labels),
             yscale=:log10, yticks=(yticks_pos, yticks_labels),
             ylims=(ylims_bottom, ylims_top),
             fontsize=fontsize, tickfontsize=fontsize,
             guidefontsize=fontsize, legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm, top_margin=14Plots.mm,
             linewidth=2.5, markersize=7, markerstrokewidth=0.1)
  
  color_idx = 1
  for order in sketching_orders
    # For order 1, handle both recursive and non-recursive cases
    if order == 1
      # Check if is_recursive column exists
      has_recursive_col = :is_recursive in propertynames(df)
      
      # Recursive case (plot first to appear first in legend)
      if has_recursive_col
        sub_recursive = df[(df.sketching_order .== order) .& (df.is_recursive .== true), :]
        if !isempty(sub_recursive)
          idx_sorted = sortperm(sub_recursive.n_vertices)
          x_vals = sub_recursive.n_vertices[idx_sorted]
          y_vals = sub_recursive.runtime_mean[idx_sorted]
          y_err = sub_recursive.runtime_std[idx_sorted]
          
          plot!(plt, x_vals, y_vals, yerror=y_err,
                marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order) \\, \\mathrm{(recursive)}"), 
                color=color_idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1, linestyle=:dot)
          color_idx += 1
        end
      end
      
      # Non-recursive case (plot second)
      if has_recursive_col
        sub_non_recursive = df[(df.sketching_order .== order) .& (df.is_recursive .== false), :]
      else
        # Backward compatibility: if column doesn't exist, all order 1 data is non-recursive
        sub_non_recursive = df[df.sketching_order .== order, :]
      end
      if !isempty(sub_non_recursive)
        idx_sorted = sortperm(sub_non_recursive.n_vertices)
        x_vals = sub_non_recursive.n_vertices[idx_sorted]
        y_vals = sub_non_recursive.runtime_mean[idx_sorted]
        y_err = sub_non_recursive.runtime_std[idx_sorted]
        
        plot!(plt, x_vals, y_vals, yerror=y_err,
              marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order)"), 
              color=color_idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1, linestyle=:solid,
              errorevery=1, capsize=3, capthickness=1.5)
        color_idx += 1
      end
    else
      # For other orders, use standard behavior (always non-recursive)
      # Filter to non-recursive if column exists, otherwise take all
      if :is_recursive in propertynames(df)
        sub = df[(df.sketching_order .== order) .& (df.is_recursive .== false), :]
      else
        sub = df[df.sketching_order .== order, :]
      end
      if !isempty(sub)
        idx_sorted = sortperm(sub.n_vertices)
        x_vals = sub.n_vertices[idx_sorted]
        y_vals = sub.runtime_mean[idx_sorted]
        y_err = sub.runtime_std[idx_sorted]
        
        plot!(plt, x_vals, y_vals, yerror=y_err,
              marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order)"), 
              color=color_idx, linewidth=2.5, markersize=7, markerstrokewidth=0.1,
              errorevery=1, capsize=3, capthickness=1.5)
        color_idx += 1
      end
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

if abspath(PROGRAM_FILE) == @__FILE__
  println("="^60)
  println("Running sketching window size benchmark...")
  println("="^60)
  df, plt = run_benchmark(max_n=12, β=1.0, sketching_orders=[1, 2, 3],
                          n_runs=1, base_seed=1234)
  
  println("\n" * "="^60)
  println("Benchmark complete!")
  println("  Plots saved:")
  println("    - system_size_error_vs_n.pdf")
  println("    - system_size_runtime_vs_n.pdf")
  println("    - system_size_combined.pdf")
  println("="^60)
end

