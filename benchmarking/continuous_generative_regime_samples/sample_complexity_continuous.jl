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
    # Warm-up run to ensure everything is compiled
    cttns_recov_warmup = deepcopy(cttns)
    CoreDeterminingEquations.compute_Gks!(f_ground_truth, cttns_recov_warmup;
                                         sketching_kwargs=sketching_kwargs,
                                         normalize_Gks=false)
    
    # Now do the actual benchmark with proper settings
    # BenchmarkTools automatically does warm-up, but we've also done a manual warm-up above
    # Use samples=3 for multiple measurements to get more reliable results
    cttns_recov_bench = deepcopy(cttns)
    bench_result = @benchmark CoreDeterminingEquations.compute_Gks!(
      $f_ground_truth, $cttns_recov_bench;
      sketching_kwargs=$sketching_kwargs,
      normalize_Gks=false
    ) samples=3 evals=1
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
Run benchmark for first order Markov case with d=5, sweeping over sample numbers.
"""
function run_benchmark_continuous(; β::Real = 1.0,
                                  recompute::Bool = true,
                                  n_runs::Int = 1,
                                  base_seed::Int = 1234,
                                  initialization_mode::Symbol = :fixed_modes,
                                  n_active_modes::Int = 10,
                                  T::Real = 1.0,
                                  basis_expansion_order::Int = 1,
                                  sample_exponents::Vector{Float64} = [3.0, 3.5, 4.0, 4.5, 5.0],
                                  track_runtime::Bool = false)
  mode_label = initialization_mode == :random ? "random" : "fixed_$(n_active_modes)_modes"
  results_path = joinpath(@__DIR__, "sample_complexity_continuous_$(mode_label)_results.csv")
  set_warn_order(6)  # d=5, so max_n+1 = 6
  
  # Auto-reload if CSV exists (always use existing data if available)
  csv_exists = isfile(results_path)
  df = nothing
  
  if csv_exists
    println("Loading existing results from: $results_path")
    try
      df = CSV.read(results_path, DataFrame)
      # Handle backward compatibility: if is_recursive column doesn't exist, add it (default to false)
      if !(:is_recursive in propertynames(df))
        df.is_recursive = false
      end
      # Handle backward compatibility: if n_samples or sample_exponent don't exist, we need to recompute
      if !(:n_samples in propertynames(df)) || !(:sample_exponent in propertynames(df))
        println("WARNING: Existing CSV missing n_samples or sample_exponent columns. Recomputing...")
        df = nothing  # Clear df to force recompute
        csv_exists = false  # Force recompute
      end
    catch e
      println("WARNING: Error reading CSV file: $e. Recomputing...")
      df = nothing
      csv_exists = false
    end
  end
  
  if !csv_exists && recompute
    rows = Vector{NamedTuple}()
    
    # Only run for first order Markov (sketching_order = 1) and d=5 (n_vertices = 5)
    # Only non-recursive mode
    sketching_order = 1
    n_vertices = 5
    is_recursive = false
    
    # Global warm-up: do one full run before starting benchmarks to ensure everything is compiled
    if track_runtime
      println("Performing global warm-up run to ensure precompilation...")
      warmup_seed = base_seed - 1  # Use a different seed for warm-up
      warmup_n_samples = Int(round(10.0^sample_exponents[1]))  # Use first sample size
      _ = benchmark_single_continuous(sketching_order, n_vertices;
                                      β=β, seed=warmup_seed, enforce_non_recursive=true,
                                      initialization_mode=initialization_mode,
                                      n_active_modes=n_active_modes,
                                      T=T,
                                      basis_expansion_order=basis_expansion_order,
                                      n_samples=warmup_n_samples,
                                      track_runtime=false)  # Don't track runtime for warm-up
      println("Warm-up complete.\n")
    end
    
    # Loop over sample numbers
    for sample_exp in sample_exponents
      n_samples = Int(round(10.0^sample_exp))
      println("Running benchmark with n_samples = $n_samples (10^$sample_exp)")
      
      max_rel_errors = Float64[]
      mean_rel_errors = Float64[]
      max_abs_diffs = Float64[]
      runtimes = Float64[]
      all_pointwise_rel_errors = Float64[]  # Collect all pointwise errors across all runs
      
      for run_idx in 1:n_runs
        seed = base_seed + run_idx - 1
        
        enforce_non_recursive = true  # Non-recursive mode only
        
        max_rel_error, mean_rel_error, max_abs_diff, runtime_seconds, pointwise_rel_errors =
          benchmark_single_continuous(sketching_order, n_vertices;
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
        n_vertices = n_vertices,
        n_samples = n_samples,
        sample_exponent = sample_exp,
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
    
    df = DataFrame(rows)
    CSV.write(results_path, df)
    println("Results saved to: $results_path")
  elseif !csv_exists && !recompute
    error("CSV file not found. Set recompute=true to generate data.")
  end
  
  # Filter to desired β and initialization mode (df should be defined at this point)
  if df === nothing
    error("No data available. CSV file not found and recompute=false.")
  end
  df = df[df.beta .== β .&& df.initialization_mode .== string(initialization_mode), :]
  
  return df
end

"""
Create error plot showing mean relative error vs number of samples.
"""
function create_error_plot_continuous(df; fontsize=18, label_prefix="(a)")
  # Check if required columns exist
  if !(:sample_exponent in propertynames(df))
    error("DataFrame missing 'sample_exponent' column. Please recompute results.")
  end
  
  # Get y-axis range - include max values to account for error bars extending upward
  all_errors_mean = df.mean_rel_error_mean
  all_errors_max = df.max_rel_error_max
  all_errors_min = df.min_rel_error_mean
  
  # Filter out invalid values
  all_errors_mean = filter(e -> isfinite(e) && !isnan(e) && !isinf(e) && e > 0, all_errors_mean)
  all_errors_max = filter(e -> isfinite(e) && !isnan(e) && !isinf(e) && e > 0, all_errors_max)
  all_errors_min = filter(e -> isfinite(e) && !isnan(e) && !isinf(e) && e > 0, all_errors_min)
  
  # Fixed y-axis limits: clip to 10^-10 to 10^2
  ylims_bottom = 1e-10
  ylims_top = 1e2
  
  # Generate y-axis ticks based on the clipped range (10^-10 to 10^2)
  ylims_bottom_exp = -10
  ylims_top_exp = 2
  
  y_ticks_pos = Float64[]
  y_ticks_labels = LaTeXString[]
  
  # Generate ticks: only even exponents in the range -10 to 2
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
  
  # Generate x-axis ticks with LaTeX strings (sample numbers)
  sample_exps = sort(unique(df.sample_exponent))
  xticks_pos = [10.0^exp for exp in sample_exps]
  xticks_labels = [latexstring("10^{$(exp)}") for exp in sample_exps]
  
  plt = plot(xlabel=L"\mathrm{Number~of~samples~}N", ylabel="",
             title=L"\mathrm{(a)~Mean~relative~error}",
             legend=:topright,
             xticks=(xticks_pos, xticks_labels),
             yticks=(y_ticks_pos, y_ticks_labels),
             xscale=:log10,
             yscale=:log10,  # Log scale
             ylims=(ylims_bottom, ylims_top),
             fontsize=fontsize, tickfontsize=fontsize, 
             guidefontsize=fontsize, legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm, top_margin=14Plots.mm,
             linewidth=2.5, markersize=7)
  
  # Sort by sample exponent
  idx_sorted = sortperm(df.sample_exponent)
  x_vals = [10.0^exp for exp in df.sample_exponent[idx_sorted]]
  y_vals_raw = Vector{Float64}(df.mean_rel_error_mean[idx_sorted])
  min_vals_raw = Vector{Float64}(df.min_rel_error_mean[idx_sorted])
  max_vals_raw = Vector{Float64}(df.max_rel_error_max[idx_sorted])
  
  # Filter out invalid values - need valid mean, min, and max for error bars
  valid_idx = [i for i in 1:length(y_vals_raw) 
                if isfinite(y_vals_raw[i]) && !isnan(y_vals_raw[i]) && !isinf(y_vals_raw[i]) && y_vals_raw[i] >= 0
                && isfinite(min_vals_raw[i]) && !isnan(min_vals_raw[i]) && !isinf(min_vals_raw[i]) && min_vals_raw[i] > 0
                && isfinite(max_vals_raw[i]) && !isnan(max_vals_raw[i]) && !isinf(max_vals_raw[i]) && max_vals_raw[i] > 0]
  x_vals = x_vals[valid_idx]
  y_vals = y_vals_raw[valid_idx]
  zero_floor_exp = ylims_bottom_exp % 2 == 0 ? ylims_bottom_exp : ylims_bottom_exp + 1
  zero_floor = 10.0^zero_floor_exp
  y_vals = ifelse.(y_vals .== 0, zero_floor, y_vals)
  y_vals = max.(y_vals, ylims_bottom)
  min_vals = min_vals_raw[valid_idx]
  max_vals = max_vals_raw[valid_idx]
  
  # Ensure min <= mean <= max (sanity check)
  for i in 1:length(y_vals)
    if min_vals[i] > y_vals[i]
      min_vals[i] = max(1e-20, y_vals[i] * 0.1)  # Fallback: 10% of mean
    end
    if max_vals[i] < y_vals[i]
      max_vals[i] = y_vals[i] * 10.0  # Fallback: 10x the mean
    end
  end
  
  # Use blue color to match other plots
  plot_color = :blue
  
  # Compute error bars: lower = mean - min, upper = max - mean
  # This shows the full range from minimum to maximum relative error (not standard deviation)
  y_err_lower = [max(0.0, y_vals[i] - min_vals[i]) for i in 1:length(y_vals)]
  y_err_upper = [max(0.0, max_vals[i] - y_vals[i]) for i in 1:length(y_vals)]
  plot!(plt, x_vals, y_vals, yerror=(y_err_lower, y_err_upper),
        marker=:o, label="", 
        color=plot_color, linecolor=plot_color, markercolor=plot_color,
        linewidth=2.5, markersize=7,
        capsize=3, capthickness=1.5)
  
  return plt
end

"""
Create runtime plot showing runtime vs number of samples.
"""
function create_runtime_plot_continuous(df; fontsize=18, label_prefix="(b)")
  # Check if required columns exist
  if !(:sample_exponent in propertynames(df))
    error("DataFrame missing 'sample_exponent' column. Please recompute results.")
  end
  
  # Check if runtime data exists
  if !(:runtime_mean in propertynames(df)) || all(ismissing.(df.runtime_mean))
    println("WARNING: No runtime data available - skipping runtime plot")
    return nothing
  end
  
  # Get y-axis range to generate ticks
  all_runtimes = filter(!ismissing, df.runtime_mean)
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
  
  # Generate ticks based on data range
  data_min_exp = floor(Int, log10(ylims_bottom))
  data_max_exp = ceil(Int, log10(ylims_top))
  
  # Generate all ticks in the range, including half powers
  yticks_pos = Float64[]
  yticks_labels = LaTeXString[]
  
  for exp in data_min_exp:data_max_exp
    # Add main tick
    tick_val = 10.0^exp
    if tick_val >= ylims_bottom && tick_val <= ylims_top
      push!(yticks_pos, tick_val)
      if exp == 0
        push!(yticks_labels, latexstring("1"))
      else
        push!(yticks_labels, latexstring("10^{$exp}"))
      end
    end
    
    # Add half-power ticks if they fall within the range
    half_exp = exp + 0.5
    half_tick_val = 10.0^half_exp
    if half_tick_val >= ylims_bottom && half_tick_val <= ylims_top
      push!(yticks_pos, half_tick_val)
      push!(yticks_labels, latexstring("10^{$(half_exp)}"))
    end
  end
  
  # Sort ticks by position
  sorted_idx = sortperm(yticks_pos)
  yticks_pos = yticks_pos[sorted_idx]
  yticks_labels = yticks_labels[sorted_idx]
  
  # Generate x-axis ticks with LaTeX strings (sample numbers)
  sample_exps = sort(unique(df.sample_exponent))
  xticks_pos = [10.0^exp for exp in sample_exps]
  xticks_labels = [latexstring("10^{$(exp)}") for exp in sample_exps]
  
  plt = plot(xlabel=L"\mathrm{Number~of~samples~}N", ylabel="",
             title=L"\mathrm{(b)~Runtime~(seconds)}",
             legend=:topright,
             xticks=(xticks_pos, xticks_labels),
             yscale=:log10, yticks=(yticks_pos, yticks_labels),
             xscale=:log10,
             ylims=(ylims_bottom, ylims_top),
             fontsize=fontsize, tickfontsize=fontsize,
             guidefontsize=fontsize, legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm, top_margin=14Plots.mm,
             linewidth=2.5, markersize=7)
  
  # Sort by sample exponent
  idx_sorted = sortperm(df.sample_exponent)
  x_vals = [10.0^exp for exp in df.sample_exponent[idx_sorted]]
  y_vals = df.runtime_mean[idx_sorted]
  
  # Filter out missing values
  valid_idx = [i for i in 1:length(y_vals) if !ismissing(y_vals[i]) && isfinite(y_vals[i])]
  if !isempty(valid_idx)
    x_vals = x_vals[valid_idx]
    y_vals = y_vals[valid_idx]
    plot!(plt, x_vals, y_vals,
          marker=:o, label="", 
          color=:blue, linewidth=2.5, markersize=7)
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
  println("Running continuous sample complexity benchmark...")
  println("First order Markov case, d=5, non-recursive")
  println("="^60)

  
  # Run for random initialization (general case)
  println("\n--- Random Initialization (General Case) ---")
  df_random = run_benchmark_continuous(β=1.0,
                                        recompute=true, n_runs=1, base_seed=1234,
                                        initialization_mode=:random,
                                        T=1.0, basis_expansion_order=2,
                                        sample_exponents=[3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                                        track_runtime=true)
  
  # Create error plot
  plt_error = create_error_plot_continuous(df_random; fontsize=18, label_prefix="(a)")
  
  # Create runtime plot
  plt_runtime = create_runtime_plot_continuous(df_random; fontsize=18, label_prefix="(b)")
  
  # Create and save combined plot
  plt_combined = plot_combined_continuous(plt_error, plt_runtime; fontsize=18)
  mode_label = "random"
  output_pdf_combined = joinpath(@__DIR__, "sample_complexity_continuous_$(mode_label)_combined.pdf")
  savefig(plt_combined, output_pdf_combined)
  println("Combined plot saved to: $output_pdf_combined")
  
  println("\n" * "="^60)
  println("Benchmark complete!")
  println("="^60)
end

