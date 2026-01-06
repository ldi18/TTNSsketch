include(joinpath(@__DIR__, "..", "..", "TTNSsketch.jl", "src", "TTNSsketch.jl"))
using .TTNSsketch
using .TTNSsketch.ExampleTopologies
using .TTNSsketch.GraphicalModels
using .TTNSsketch.CoreDeterminingEquations
using .TTNSsketch.ErrorReporting: report_errors
using .TTNSsketch.Sketching
using .TTNSsketch.ContinuousVariableEmbedding
using Printf
using CSV
using DataFrames
using Plots
using LaTeXStrings
using Random
using Statistics
using BenchmarkTools
using ITensors: set_warn_order
using NamedGraphs: edges, NamedEdge
using Plots.PlotMeasures

"""
Create sketching kwargs for Markov sketching.
"""
function create_sketching_kwargs(order::Int, seed::Int; enforce_non_recursive::Bool = false)
  kwargs = Dict{Symbol, Any}(
    :sketching_type => Sketching.Markov,
    :order => order,
    :seed => seed,
    :use_max_order_only => false,
    :connected_only_expansion => false,
    :truncate_reweight_highest => false,
    :theta_sketch_function => Sketching.ThetaSketchingFuncs.theta_ls,
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
function benchmark_single_continuous(sketching_order::Int, n_vertices::Int;
                                     β::Real = 1.0,
                                     seed::Int = 1234,
                                     enforce_non_recursive::Bool = false,
                                     initialization_mode::Symbol = :fixed_modes,
                                     n_active_modes::Int = 10,
                                     T::Real = 1.0,
                                     basis_expansion_order::Int = 1,
                                     n_samples::Int = 50000,
                                     track_runtime::Bool = false)
  # Set up continuous topology
  # Note: legendre_basis expects :a and :b for domain bounds, with defaults a=0, b=1
  # The domain will be [0, T], so we set a=0, b=T
  local_basis_kwargs = Dict{Symbol, Any}(
    :local_basis_func_set => ContinuousVariableEmbedding.legendre_basis,
    :local_basis_func_kwargs => Dict{Symbol, Any}(:a => 0.0, :b => T, :basis_expansion_order => basis_expansion_order)
  )
  cttns = ExampleTopologies.Linear(n_vertices; continuous=true, local_basis_kwargs=local_basis_kwargs)
  
  # Create ground truth model
  if initialization_mode == :random
    model_ground_truth = GraphicalModels.random_cGraphicalModel(cttns; seed=seed)
  elseif initialization_mode == :fixed_modes
    error("Fixed modes initialization not implemented. Please provide your own implementation.")
  else
    error("Unknown initialization_mode: $initialization_mode. Use :random or :fixed_modes")
  end
  
  # Generate sample points and evaluate ground truth function
  d = size(collect(cttns.vertex_to_input_pos_map))[1]
  Random.seed!(seed)  # Use same seed as model for consistency
  xs = [Tuple(T * rand(d)) for _ in 1:n_samples]
  f_ground_truth = Dict(x => TTNSsketch.evaluate(model_ground_truth.ttns, x) for x in xs)
  
  # Recover using sketching
  sketching_kwargs = create_sketching_kwargs(sketching_order, seed; enforce_non_recursive=enforce_non_recursive)
  runtime_seconds = missing
  if track_runtime
    cttns_recov_bench = deepcopy(cttns)
    bench_result = @benchmark CoreDeterminingEquations.compute_Gks!(
      $f_ground_truth, $cttns_recov_bench;
      sketching_kwargs=$sketching_kwargs,
      normalize_Gks=false
    ) evals=1
    runtime_seconds = mean(bench_result.times) / 1e9
  end
  
  cttns_recov = deepcopy(cttns)
  CoreDeterminingEquations.compute_Gks!(f_ground_truth, cttns_recov;
                                       sketching_kwargs=sketching_kwargs,
                                       normalize_Gks=false)
  
  # Compute pointwise relative errors
  pointwise_rel_errors = Float64[]
  max_rel_error = 0.0
  mean_rel_error = 0.0
  max_abs_diff = 0.0
  
  f_vals_ground_truth = [f_ground_truth[x] for x in xs]
  max_f_val = maximum(f_vals_ground_truth)
  
  for x in xs
    f_ref = f_ground_truth[x]
    f_recov = TTNSsketch.evaluate(cttns_recov, x)
    abs_diff = abs(f_ref - f_recov)
    # Use relative error normalized by max function value (as in cttns_high_level_error)
    rel_error = max_f_val > 0 ? abs_diff / max_f_val : (abs_diff > 0 ? Inf : 0.0)
    
    push!(pointwise_rel_errors, rel_error)
    max_rel_error = max(max_rel_error, rel_error)
    mean_rel_error += rel_error
    max_abs_diff = max(max_abs_diff, abs_diff)
  end
  mean_rel_error /= length(xs)
  
  return max_rel_error, mean_rel_error, max_abs_diff, runtime_seconds, pointwise_rel_errors
end

"""
Sweep over system sizes and sketching orders, then plot high-level error vs system size.
"""
function run_benchmark_continuous(; max_n::Int = 12,
                                  β::Real = 1.0,
                                  sketching_orders::Vector{Int} = [1],
                                  recompute::Bool = true,
                                  n_runs::Int = 1,
                                  base_seed::Int = 1234,
                                  initialization_mode::Symbol = :fixed_modes,
                                  n_active_modes::Int = 10,
                                  T::Real = 1.0,
                                  basis_expansion_order::Int = 1,
                                  n_samples::Int = 50000,
                                  track_runtime::Bool = false)
  mode_label = initialization_mode == :random ? "random" : "fixed_$(n_active_modes)_modes"
  results_path = joinpath(@__DIR__, "system_size_continuous_$(mode_label)_results.csv")
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
  elseif recompute
    rows = Vector{NamedTuple}()
    
    # Run benchmarks using BenchmarkTools for accurate timing (handles warm-up automatically)
    for sketching_order in sketching_orders
      # For order 1, run both recursive and non-recursive cases
      recursive_cases = sketching_order == 1 ? [false, true] : [false]
      
      for is_recursive in recursive_cases
        for n in 1:max_n
          max_rel_errors = Float64[]
          mean_rel_errors = Float64[]
          max_abs_diffs = Float64[]
          runtimes = Float64[]
          all_pointwise_rel_errors = Float64[]  # Collect all pointwise errors across all runs
          
          for run_idx in 1:n_runs
            seed = base_seed + run_idx - 1
            
            enforce_non_recursive = !is_recursive  # If recursive, don't enforce non-recursive
            
            max_rel_error, mean_rel_error, max_abs_diff, runtime_seconds, pointwise_rel_errors =
              benchmark_single_continuous(sketching_order, n;
                                          β=β, seed=seed, enforce_non_recursive=enforce_non_recursive,
                                          initialization_mode=initialization_mode,
                                          n_active_modes=n_active_modes,
                                          T=T,
                                          basis_expansion_order=basis_expansion_order,
                                          n_samples=n_samples,
                                          track_runtime=track_runtime)
            
            # Collect all pointwise errors
            append!(all_pointwise_rel_errors, pointwise_rel_errors)
            
            push!(max_rel_errors, max_rel_error)
            push!(mean_rel_errors, mean_rel_error)
            push!(max_abs_diffs, max_abs_diff)
            if track_runtime
              push!(runtimes, runtime_seconds)
            end
          end
          
          # Compute mean, min, max, and std across all pointwise errors (not across runs)
          pointwise_error_mean = mean(all_pointwise_rel_errors)
          pointwise_error_std = std(all_pointwise_rel_errors)
          
          # Compute min - filter out invalid values first
          valid_errors = filter(e -> isfinite(e) && !isnan(e) && !isinf(e) && e > 0, all_pointwise_rel_errors)
          if !isempty(valid_errors)
            pointwise_error_min = minimum(valid_errors)
          else
            # Fallback: estimate min from mean and std (assuming normal distribution, use mean - 2*std)
            pointwise_error_min = max(1e-20, pointwise_error_mean - 2.0 * pointwise_error_std)
          end
          
          # Compute max - filter out invalid values first
          if !isempty(valid_errors)
            pointwise_error_max = maximum(valid_errors)
          else
            # Fallback: estimate max from mean and std
            pointwise_error_max = pointwise_error_mean + 2.0 * pointwise_error_std
          end
          
          # Ensure min is valid and positive
          if !isfinite(pointwise_error_min) || isnan(pointwise_error_min) || isinf(pointwise_error_min) || pointwise_error_min <= 0
            pointwise_error_min = max(1e-20, pointwise_error_mean - 2.0 * pointwise_error_std)
          end
          
          # Handle runtime statistics
          if track_runtime && !isempty(runtimes)
            runtime_mean_val = mean(runtimes)
            runtime_std_val = std(runtimes)
            runtime_min_val = minimum(runtimes)
            runtime_max_val = maximum(runtimes)
          else
            runtime_mean_val = missing
            runtime_std_val = missing
            runtime_min_val = missing
            runtime_max_val = missing
          end
          
          push!(rows, (
            sketching_order = sketching_order,
            is_recursive = is_recursive,
            n_vertices = n,
            beta = β,
            initialization_mode = string(initialization_mode),
            n_active_modes = initialization_mode == :fixed_modes ? n_active_modes : -1,
            max_rel_error_mean = mean(max_rel_errors),
            max_rel_error_std = std(max_rel_errors),
            mean_rel_error_mean = pointwise_error_mean,  # Mean across all points
            mean_rel_error_std = pointwise_error_std,   # Std across all points
            min_rel_error_mean = pointwise_error_min,   # Min across all points
            max_rel_error_max = pointwise_error_max,    # Max across all points (for error bars)
            max_abs_diff_mean = mean(max_abs_diffs),
            max_abs_diff_std = std(max_abs_diffs),
            runtime_mean = runtime_mean_val,
            runtime_std = runtime_std_val,
            runtime_min = runtime_min_val,
            runtime_max = runtime_max_val,
          ))
        end
      end
    end
    
    df = DataFrame(rows)
    CSV.write(results_path, df)
    println("Results saved to: $results_path")
  else
    error("CSV file not found. Set recompute=true to generate data.")
  end
  
  # Filter to desired β and initialization mode
  df = df[df.beta .== β .&& df.initialization_mode .== string(initialization_mode), :]
  
  # Determine label prefix based on initialization mode
  # Fixed modes uses (c) and (d), random uses (a) and (b)
  error_label = initialization_mode == :fixed_modes ? "(c)" : "(a)"
  runtime_label = initialization_mode == :fixed_modes ? "(d)" : "(b)"
  
  # Create error plot
  plt_error = create_error_plot_continuous(df, sketching_orders; fontsize=18, label_prefix=error_label)
  
  # Create runtime plot
  plt_runtime = create_runtime_plot_continuous(df, sketching_orders; fontsize=18, label_prefix=runtime_label)
  
  # Create and save combined plot only
  plt_combined = plot_combined_continuous(plt_error, plt_runtime; fontsize=18)
  output_pdf_combined = joinpath(@__DIR__, "system_size_continuous_$(mode_label)_combined.pdf")
  savefig(plt_combined, output_pdf_combined)
  println("Combined plot saved to: $output_pdf_combined")
  
  return df, plt_combined
end

"""
Create error plot showing high-level error vs system size (linear scale with scalebreak at 10).
"""
function create_error_plot_continuous(df, sketching_orders; fontsize=18, label_prefix="(a)")
  # Get y-axis range - include max values to account for error bars extending upward
  all_errors_mean = vcat([df[df.sketching_order .== order, :].mean_rel_error_mean for order in sketching_orders]...)
  all_errors_max = vcat([df[df.sketching_order .== order, :].max_rel_error_max for order in sketching_orders]...)
  all_errors_min = vcat([df[df.sketching_order .== order, :].min_rel_error_mean for order in sketching_orders]...)
  
  # Filter out invalid values
  all_errors_mean = filter(e -> isfinite(e) && !isnan(e) && !isinf(e) && e > 0, all_errors_mean)
  all_errors_max = filter(e -> isfinite(e) && !isnan(e) && !isinf(e) && e > 0, all_errors_max)
  all_errors_min = filter(e -> isfinite(e) && !isnan(e) && !isinf(e) && e > 0, all_errors_min)
  
  min_error = isempty(all_errors_min) ? (isempty(all_errors_mean) ? 1e-6 : minimum(all_errors_mean)) : minimum(all_errors_mean)
  max_error = isempty(all_errors_max) ? (isempty(all_errors_mean) ? 1.0 : maximum(all_errors_mean)) : maximum(all_errors_max)
  
  # Calculate y-axis limits with padding to accommodate error bars
  ylims_bottom = 1e-17  # Fixed lower limit to show all error bars
  max_error_exp = isfinite(log10(max_error)) ? ceil(Int, log10(max_error)) : 2
  ylims_top = 1e4
  
  # Generate y-axis ticks based on the full displayed range (ylims_bottom to ylims_top)
  ylims_bottom_exp = isfinite(log10(ylims_bottom)) ? floor(Int, log10(ylims_bottom)) : -17
  ylims_top_exp = isfinite(log10(ylims_top)) ? ceil(Int, log10(ylims_top)) : 2
  
  y_ticks_pos = Float64[]
  y_ticks_labels = LaTeXString[]
  
  # Generate ticks: only even exponents in the full displayed range
  for exp in ylims_bottom_exp:ylims_top_exp
    if exp % 2 == 0  # Only even exponents
      tick_val = 10.0^exp
      push!(y_ticks_pos, tick_val)
      if exp == 0
        push!(y_ticks_labels, latexstring("1"))
      else
        push!(y_ticks_labels, latexstring("10^{$exp}"))
      end
    end
  end
  
  # Generate x-axis ticks with LaTeX strings
  n_vals = sort(unique(df.n_vertices))
  xticks_pos = Float64.(n_vals)
  xticks_labels = [latexstring("$val") for val in n_vals]
  
  plt = plot(xlabel=L"\mathrm{System~size~}d",
             ylabel=latexstring("\\mathrm{Mean~rel.~error}"),
             title=L"\mathrm{(a)~Mean~relative~error~vs.~System~size~}d",
             legend=:bottomright,
             xticks=(xticks_pos, xticks_labels),
             yticks=(y_ticks_pos, y_ticks_labels),
             yscale=:log10,  # Log scale
             ylims=(ylims_bottom, ylims_top),
             fontsize=fontsize, tickfontsize=fontsize, 
             guidefontsize=fontsize, legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm, top_margin=14Plots.mm,
             linewidth=2.5, markersize=7)
  
  color_idx = 1
  for order in sketching_orders
    # For error plot, only show non-recursive data
    # Filter to non-recursive if column exists, otherwise take all
    if :is_recursive in propertynames(df)
      sub = df[(df.sketching_order .== order) .& (df.is_recursive .== false), :]
    else
      # Backward compatibility: if column doesn't exist, all data is non-recursive
      sub = df[df.sketching_order .== order, :]
    end
    if !isempty(sub)
      idx_sorted = sortperm(sub.n_vertices)
      x_vals = sub.n_vertices[idx_sorted]
      y_vals_raw = Vector{Float64}(sub.mean_rel_error_mean[idx_sorted])
      # Get min and max for error bars (from min to max relative error)
      min_vals_raw = Vector{Float64}(sub.min_rel_error_mean[idx_sorted])
      max_vals_raw = Vector{Float64}(sub.max_rel_error_max[idx_sorted])
      std_vals_raw = hasproperty(sub, :mean_rel_error_std) ? Vector{Float64}(sub.mean_rel_error_std[idx_sorted]) : fill(0.0, length(y_vals_raw))
      
      # Filter out invalid values
      valid_idx = [i for i in 1:length(y_vals_raw) if isfinite(y_vals_raw[i]) && !isnan(y_vals_raw[i]) && !isinf(y_vals_raw[i]) && y_vals_raw[i] >= 0]
      x_vals = x_vals[valid_idx]
      y_vals = y_vals_raw[valid_idx]
      zero_floor_exp = ylims_bottom_exp % 2 == 0 ? ylims_bottom_exp : ylims_bottom_exp + 1
      zero_floor = 10.0^zero_floor_exp
      y_vals = ifelse.(y_vals .== 0, zero_floor, y_vals)
      y_vals = max.(y_vals, ylims_bottom)
      min_vals = min_vals_raw[valid_idx]
      max_vals = max_vals_raw[valid_idx]
      std_vals = std_vals_raw[valid_idx]
      
      # Fix invalid min values: compute from mean and std if needed
      for i in 1:length(y_vals)
        if !isfinite(min_vals[i]) || isnan(min_vals[i]) || isinf(min_vals[i]) || min_vals[i] <= 0
          # Estimate min from mean and std (mean - 2*std, but ensure positive)
          min_vals[i] = max(1e-20, y_vals[i] - 2.0 * std_vals[i])
        end
      end
      
      # Get the actual color from the palette
      plot_color = palette(:default)[color_idx]
      
      # Compute asymmetric error bars: lower = mean - min, upper = max - mean
      # Filter to ensure max is valid
      valid_err_idx = [i for i in 1:length(y_vals) if isfinite(max_vals[i]) && !isnan(max_vals[i]) && !isinf(max_vals[i])]
      if length(valid_err_idx) == length(y_vals)
        y_err_lower = [max(0.0, y_vals[i] - min_vals[i]) for i in 1:length(y_vals)]
        y_err_upper = [max(0.0, max_vals[i] - y_vals[i]) for i in 1:length(y_vals)]
        plot!(plt, x_vals, y_vals, yerror=(y_err_lower, y_err_upper),
              marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order)"), 
              color=plot_color, linecolor=plot_color, markercolor=plot_color,
              linewidth=2.5, markersize=7,
              capsize=3, capthickness=1.5)
      else
        # If some min/max values are invalid, plot without error bars for those points
        x_vals_valid = x_vals[valid_err_idx]
        y_vals_valid = y_vals[valid_err_idx]
        min_vals_valid = min_vals[valid_err_idx]
        max_vals_valid = max_vals[valid_err_idx]
        y_err_lower = [max(0.0, y_vals_valid[i] - min_vals_valid[i]) for i in 1:length(y_vals_valid)]
        y_err_upper = [max(0.0, max_vals_valid[i] - y_vals_valid[i]) for i in 1:length(y_vals_valid)]
        plot!(plt, x_vals_valid, y_vals_valid, yerror=(y_err_lower, y_err_upper),
              marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order)"), 
              color=plot_color, linecolor=plot_color, markercolor=plot_color,
              linewidth=2.5, markersize=7,
              capsize=3, capthickness=1.5)
      end
      color_idx += 1
    end
  end
  
  return plt
end

"""
Create runtime plot showing runtime vs system size (log scale).
"""
function create_runtime_plot_continuous(df, sketching_orders; fontsize=18, label_prefix="(b)")
  # Check if runtime data exists
  if !(:runtime_mean in propertynames(df)) || all(ismissing.(df.runtime_mean))
    println("WARNING: No runtime data available - skipping runtime plot")
    return nothing
  end
  
  # Get y-axis range to generate ticks
  all_runtimes = vcat([df[df.sketching_order .== order, :].runtime_mean for order in sketching_orders]...)
  # Filter out missing values
  all_runtimes = filter(!ismissing, all_runtimes)
  if isempty(all_runtimes)
    println("WARNING: No valid runtime data after filtering - skipping runtime plot")
    return nothing
  end
  min_runtime = minimum(all_runtimes)
  max_runtime = maximum(all_runtimes)
  min_runtime = max(min_runtime, 1e-20)
  
  # Calculate y-axis limits
  ylims_bottom = max(min_runtime * 0.8, 1e-20)
  ylims_top = max_runtime * 1.3
  
  # Generate ticks based on data range - show all ticks in the range
  data_min_exp = -1
  data_max_exp = 2
  min_tick_val = 10.0^-1
  max_tick_val = 10.0^2.5
  
  # Generate all ticks in the range, including half powers
  yticks_pos = Float64[]
  yticks_labels = LaTeXString[]
  
  for exp in data_min_exp:data_max_exp
    # Add main tick
    tick_val = 10.0^exp
    if tick_val >= min_tick_val && tick_val <= max_tick_val
      push!(yticks_pos, tick_val)
      if exp == 0
        push!(yticks_labels, latexstring("1"))
      else
        push!(yticks_labels, latexstring("10^{$exp}"))
      end
    end
    
    # Add half-power ticks (10^-0.5, 10^0.5, 10^1.5, 10^2.5) if they fall within the range
    if exp == -1 || exp == 0 || exp == 1 || exp == 2
      half_exp = exp + 0.5
      half_tick_val = 10.0^half_exp
      if half_tick_val >= ylims_bottom && half_tick_val <= ylims_top &&
         half_tick_val >= min_tick_val && half_tick_val <= max_tick_val
        push!(yticks_pos, half_tick_val)
        # Format the exponent properly for LaTeX
        if half_exp == -0.5
          push!(yticks_labels, latexstring("10^{-0.5}"))
        elseif half_exp == 0.5
          push!(yticks_labels, latexstring("10^{0.5}"))
        elseif half_exp == 1.5
          push!(yticks_labels, latexstring("10^{1.5}"))
        elseif half_exp == 2.5
          push!(yticks_labels, latexstring("10^{2.5}"))
        end
      end
    end
  end
  
  # Sort ticks by position
  sorted_idx = sortperm(yticks_pos)
  yticks_pos = yticks_pos[sorted_idx]
  yticks_labels = yticks_labels[sorted_idx]
  
  # Generate x-axis ticks with LaTeX strings
  n_vals = sort(unique(df.n_vertices))
  xticks_pos = Float64.(n_vals)
  xticks_labels = [latexstring("$val") for val in n_vals]
  
  plt = plot(xlabel=L"\mathrm{System~size~}d",
             ylabel=latexstring("\\mathrm{Runtime~(seconds)}"),
             title=L"\mathrm{(b)~Runtime~vs.~System~size~}d",
             legend=:bottomright,
             xticks=(xticks_pos, xticks_labels),
             yscale=:log10, yticks=(yticks_pos, yticks_labels),
             ylims=(ylims_bottom, ylims_top),
             fontsize=fontsize, tickfontsize=fontsize,
             guidefontsize=fontsize, legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm, top_margin=14Plots.mm,
             linewidth=2.5, markersize=7)
  
  color_idx = 1
  for order in sketching_orders
    # For order 1, handle both recursive and non-recursive cases
    if order == 1
      # Check if is_recursive column exists
      has_recursive_col = :is_recursive in propertynames(df)
      plot_color = palette(:default)[color_idx]
      plotted_order = false
      
      # Recursive case (plot first to appear first in legend)
      if has_recursive_col
        sub_recursive = df[(df.sketching_order .== order) .& (df.is_recursive .== true), :]
        if !isempty(sub_recursive)
          idx_sorted = sortperm(sub_recursive.n_vertices)
          x_vals = sub_recursive.n_vertices[idx_sorted]
          y_vals = sub_recursive.runtime_mean[idx_sorted]
          
          # Filter out missing values
          valid_idx = [i for i in 1:length(y_vals) if !ismissing(y_vals[i]) && isfinite(y_vals[i])]
          if !isempty(valid_idx)
            x_vals = x_vals[valid_idx]
            y_vals = y_vals[valid_idx]
            plot!(plt, x_vals, y_vals,
                  marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order) \\, \\mathrm{(recursive)}"), 
                  color=plot_color, linecolor=plot_color, markercolor=plot_color,
                  linewidth=2.5, markersize=7, linestyle=:dot)
            plotted_order = true
          end
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
        
        # Filter out missing values
        valid_idx = [i for i in 1:length(y_vals) if !ismissing(y_vals[i]) && isfinite(y_vals[i])]
        if !isempty(valid_idx)
          x_vals = x_vals[valid_idx]
          y_vals = y_vals[valid_idx]
          plot!(plt, x_vals, y_vals,
                marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order)"), 
                color=plot_color, linecolor=plot_color, markercolor=plot_color,
                linewidth=2.5, markersize=7, linestyle=:solid)
          plotted_order = true
        end
      end
      if plotted_order
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
        
        # Filter out missing values
        valid_idx = [i for i in 1:length(y_vals) if !ismissing(y_vals[i]) && isfinite(y_vals[i])]
        if !isempty(valid_idx)
          x_vals = x_vals[valid_idx]
          y_vals = y_vals[valid_idx]
          plot!(plt, x_vals, y_vals,
                marker=:o, label=latexstring("n_{\\mathrm{Sketch}} = $(order)"), 
                color=color_idx, linewidth=2.5, markersize=7)
        end
        color_idx += 1
      end
    end
  end
  
  return plt
end

"""
Create combined plot with error and runtime side by side.
"""
function plot_combined_continuous(plt_error, plt_runtime; fontsize=18, height=600)
  if plt_runtime === nothing
    return plt_error
  end
  
  combined = plot(plt_error, plt_runtime, layout=(1, 2), 
                  size=(1600, height), dpi=300,
                  plot_title="",  
                  left_margin=8Plots.mm, right_margin=8Plots.mm,
                  bottom_margin=12Plots.mm, top_margin=6Plots.mm)
  
  return combined
end

if abspath(PROGRAM_FILE) == @__FILE__
  println("="^60)
  println("Running continuous sketching window size benchmark...")
  println("="^60)

  
  # Run for random initialization (general case)
  println("\n--- Random Initialization (General Case) ---")
  df_random, plt_random = run_benchmark_continuous(max_n=9, β=1.0, sketching_orders=[1, 2, 3],
                                                    recompute=true, n_runs=1, base_seed=1234,
                                                    initialization_mode=:random,
                                                    T=1.0, basis_expansion_order=2, n_samples=50000,
                                                    track_runtime=true)
  
  println("\n" * "="^60)
  println("Benchmark complete!")
  println("  Plot saved:")
  println("    - system_size_continuous_random_combined.pdf")
  println("="^60)
end
