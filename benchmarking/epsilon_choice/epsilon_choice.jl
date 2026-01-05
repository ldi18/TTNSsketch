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
Create sketching kwargs for Perturbative sketching with given epsilon.
"""
function create_sketching_kwargs(order_recovery::Int, epsilon::Float64, seed::Int; beta_dim::Int = 2)
  return Dict{Symbol, Any}(
    :sketching_type => Sketching.Perturbative,
    :order => order_recovery,
    :seed => seed,
    :epsilon => epsilon,
    :beta_dim => beta_dim,
    :use_expansion => true
  )
end

"""
Compute max relative error between ground truth and recovered TTNS using report_errors.
"""
function compute_max_relative_error(p_ground_truth::Dict, ttns_recov)
  keys_all = collect(keys(p_ground_truth))
  error_stats = report_errors(p_ground_truth, ttns_recov, keys_all; print_sections=false)
  max_rel_error = error_stats.overall.max
  return isnan(max_rel_error) ? 0.0 : max_rel_error
end

"""
Run a single benchmark for given epsilon and return max relative error and runtime.
Only uses detailed BenchmarkTools timing for epsilon = 10^(-4), uses simple timing for others.
"""
function benchmark_single(epsilon::Float64;
                          n_vertices::Int = 8,
                          order::Int = 1,
                          order_recovery::Int = 1,
                          β::Real = 1.0,
                          seed::Int = 1234,
                          reference_runtime::Union{Nothing, Float64} = nothing,
                          beta_dim::Int = 2)
  # Create topology and ground truth
  ttns = ExampleTopologies.Linear(n_vertices)
  p_ground_truth = TTNSsketch.higher_order_probability_dict(ttns; β=β, order=order)
  sketching_kwargs = create_sketching_kwargs(order_recovery, epsilon, seed; beta_dim=beta_dim)
  ttns_recov = deepcopy(ttns)

  # Only use BenchmarkTools for epsilon = 10^(-4) for accurate timing
  # Runtime of other epsilon values is nearly identical as epsilon is only a numerical factor.
  reference_epsilon = 10^(-4)
  is_reference = abs(epsilon - reference_epsilon) < 1e-10
  
  if is_reference
    # Use BenchmarkTools to measure runtime (handles warm-up automatically)
    bench_result = @benchmark CoreDeterminingEquations.compute_Gks!(
      $p_ground_truth, $ttns_recov;
      sketching_kwargs=$sketching_kwargs
    ) evals=1
    runtime_seconds = mean(bench_result.times) / 1e9
    # Get the actual result after warm-up
    ttns_recov_result = deepcopy(ttns)
    CoreDeterminingEquations.compute_Gks!(p_ground_truth, ttns_recov_result; sketching_kwargs=sketching_kwargs)
    max_rel_error = compute_max_relative_error(p_ground_truth, ttns_recov_result)
  else
    # For other epsilon values, use reference runtime if provided
    CoreDeterminingEquations.compute_Gks!(p_ground_truth, ttns_recov; sketching_kwargs=sketching_kwargs)
    if reference_runtime !== nothing
      runtime_seconds = reference_runtime
    else
      start_time = time()
      CoreDeterminingEquations.compute_Gks!(p_ground_truth, deepcopy(ttns); sketching_kwargs=sketching_kwargs)
      runtime_seconds = time() - start_time
    end
    max_rel_error = compute_max_relative_error(p_ground_truth, ttns_recov)
  end
  
  return max_rel_error, runtime_seconds
end

"""
Run epsilon sweep for multiple recovery orders and generate log-log plot with all curves.
"""
function run_epsilon_sweep(; n_vertices::Int = 7,
                            order_recovery_vals::Vector{Int} = [1, 2, 4, 6],
                            β::Real = 1.0,
                            seed::Int = 1234,
                            recompute::Bool = true,
                            output_suffix::String = "")
  # Constants
  epsilon_vals = [10^(-6), 10^(-5.5), 10^(-5), 10^(-4.5), 10^(-4), 10^(-3.5), 10^(-3), 10^(-2.5), 10^(-2), 10^(-1.5), 10^(-1), 10^(-0.5)]
  conditional_orders = [1, 2, 3]
  beta_dim_map = Dict(1 => 2, 2 => 4, 3 => 8)
  reference_epsilon = 10^(-4)
  all_data = Dict{Int, DataFrame}()
  
  for cond_order in conditional_orders
    beta_dim = beta_dim_map[cond_order]
    output_suffix_order = "$(output_suffix)_cond$(cond_order)"
    results_path_order = joinpath(@__DIR__, "epsilon_choice_results$(output_suffix_order).csv")
    
    needs_recompute = false
    if isfile(results_path_order) && !recompute
      println("Loading existing results for conditional order $cond_order from: $results_path_order")
      df_loaded = CSV.read(results_path_order, DataFrame)
      # Check if loaded data contains all requested order_recovery_vals
      loaded_orders = sort(unique(df_loaded.order_recovery))
      requested_orders = sort(order_recovery_vals)
      if loaded_orders == requested_orders
        all_data[cond_order] = df_loaded
        println("  Loaded data contains all requested sketching orders: $requested_orders")
      else
        println("  WARNING: Loaded data has sketching orders $loaded_orders, but requested $requested_orders")
        println("  Will recompute to match requested sketching orders...")
        needs_recompute = true
      end
    else
      needs_recompute = true
    end
    
    if needs_recompute
      println("Computing data for conditional order $cond_order (beta_dim=$beta_dim)...")
      rows = Vector{NamedTuple}()
      
      for order_recovery in order_recovery_vals
        # First compute reference epsilon to get runtime
        reference_error, reference_runtime = benchmark_single(reference_epsilon;
                                                               n_vertices=n_vertices,
                                                               order=cond_order,
                                                               order_recovery=order_recovery,
                                                               β=β,
                                                               seed=seed,
                                                               beta_dim=beta_dim)
        push!(rows, (order_recovery=order_recovery, epsilon=reference_epsilon, max_rel_error=reference_error, runtime=reference_runtime))
        
        # Then compute other epsilon values using reference runtime
        for epsilon in epsilon_vals
          abs(epsilon - reference_epsilon) < 1e-10 && continue
          max_rel_error, runtime_seconds = benchmark_single(epsilon;
                                           n_vertices=n_vertices,
                                           order=cond_order,
                                           order_recovery=order_recovery,
                                           β=β,
                                           seed=seed,
                                           reference_runtime=reference_runtime,
                                           beta_dim=beta_dim)
          push!(rows, (order_recovery=order_recovery, epsilon=epsilon, max_rel_error=max_rel_error, runtime=runtime_seconds))
        end
      end
      
      all_data[cond_order] = DataFrame(rows)
      CSV.write(results_path_order, all_data[cond_order])
      println("Results saved to: $results_path_order")
    end
  end
  
  # Create error plots for each conditional order
  error_plots = []
  titles = [latexstring("\\mathrm{(a)~Max.~Rel.~Error},~n_{\\mathrm{Cond}} = 1"),
            latexstring("\\mathrm{(b)~Max.~Rel.~Error},~n_{\\mathrm{Cond}} = 2"),
            latexstring("\\mathrm{(c)~Max.~Rel.~Error},~n_{\\mathrm{Cond}} = 3")]
  for (i, cond_order) in enumerate(conditional_orders)
    plt = create_error_plot(all_data[cond_order])
    plot!(plt, title=titles[i])
    push!(error_plots, plt)
  end
  plt_error1, plt_error2, plt_error3 = error_plots
  
  # Create runtime plot combining all conditional orders (for epsilon = 10^-4)
  plt_runtime = create_runtime_plot_combined(all_data)
  
  # Create 2x2 grid: error plots in upper left, upper right, lower left; runtime in lower right
  plt_combined = plot(plt_error1, plt_error2, plt_error3, plt_runtime,
                      layout=(2, 2),
                      size=(1600, 1500),
                      dpi=300,
                      plot_title="",
                      left_margin=8Plots.mm,
                      right_margin=8Plots.mm,
                      bottom_margin=12Plots.mm,
                      top_margin=6Plots.mm)
  
  output_pdf = joinpath(@__DIR__, "epsilon_choice$(output_suffix).pdf")
  savefig(plt_combined, output_pdf)
  println("Plot saved to: $output_pdf")
  
  return plt_combined, all_data
end


"""
Create error plot: epsilon vs max relative error for different recovery orders.
"""
function create_error_plot(df; fontsize=18)
  # Generate x-axis ticks for epsilon (only whole powers of 10, excluding 10^0)
  min_eps_exp = -6
  max_eps_exp = -1
  x_ticks_pos = [10.0^exp for exp in min_eps_exp:max_eps_exp]
  x_ticks_labels = [latexstring("10^{$exp}") for exp in min_eps_exp:max_eps_exp]
  
  # Cap errors at 10 for plotting
  df_plot = copy(df)
  df_plot.max_rel_error = min.(df_plot.max_rel_error, 10.0)
  
  # Generate y-axis ticks with LaTeX formatting
  # Find the range of capped errors
  min_error = minimum(df_plot.max_rel_error)
  max_error = maximum(df_plot.max_rel_error)
  
  # Generate ticks: use powers of 10, but cap at 10
  min_error_exp = isfinite(log10(min_error)) ? floor(Int, log10(min_error)) : -6
  max_error_exp = isfinite(log10(max_error)) ? ceil(Int, log10(max_error)) : 1
  
  # Ensure we include 10 if max_error is >= 10
  if max_error >= 10.0
    max_error_exp = 1  # 10^1 = 10
  end
  
  y_ticks_pos = Float64[]
  y_ticks_labels = String[]
  
  for exp in min_error_exp:max_error_exp
    tick_val = 10.0^exp
    if tick_val <= 10.0
      push!(y_ticks_pos, tick_val)
      if tick_val == 10.0 && max_error >= 10.0
        push!(y_ticks_labels, latexstring(">10"))
      else
        push!(y_ticks_labels, latexstring("10^{$exp}"))
      end
    end
  end
  
  # Create log-log plot
  plt = plot(xlabel=latexstring("\\epsilon"),
             ylabel="",
             title=latexstring("\\mathrm{(a)~Max.~Rel.~Error}~\\max_{x} |(f(x) - f_{\\mathrm{ttns}}(x))| / f(x)"),
             xscale=:log10,
             yscale=:log10,
             xticks=(x_ticks_pos, x_ticks_labels),
             yticks=(y_ticks_pos, y_ticks_labels),
             ylims=(minimum(y_ticks_pos) * 0.8, 10.0 * 1.8),
             legend=:bottomleft,
             fontsize=fontsize,
             tickfontsize=fontsize,
             guidefontsize=fontsize,
             legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm,
             top_margin=14Plots.mm,
             linewidth=2.5,
             markersize=7,
             markerstrokewidth=0.1,
             dpi=300)
  
  # Plot each recovery order as a separate curve
  order_recovery_vals_plot = sort(unique(df_plot.order_recovery))
  for (idx, order_recovery) in enumerate(order_recovery_vals_plot)
    sub_df = df_plot[df_plot.order_recovery .== order_recovery, :]
    idx_sorted = sortperm(sub_df.epsilon)
    plot!(plt, sub_df.epsilon[idx_sorted], sub_df.max_rel_error[idx_sorted],
          marker=:o,
          label=latexstring("l_{\\mathrm{max}} = $(order_recovery)"),
          color=idx,
          linewidth=2.5,
          markersize=7,
          markerstrokewidth=0.1)
  end
  
  return plt
end

"""
Create runtime plot: order_recovery vs runtime for epsilon = 10^-4, combining all conditional orders.
"""
function create_runtime_plot_combined(all_data::Dict{Int, DataFrame}; fontsize=18)
  reference_epsilon = 10^(-4)
  all_runtimes = Float64[]
  
  # Collect all runtimes from all conditional orders for epsilon = 10^-4
  for (cond_order, df) in all_data
    if !(:runtime in propertynames(df))
      continue
    end
    sub_df = df[abs.(df.epsilon .- reference_epsilon) .< 1e-10, :]
    if !isempty(sub_df)
      append!(all_runtimes, sub_df.runtime[.!ismissing.(sub_df.runtime)])
    end
  end
  
  if isempty(all_runtimes)
    println("WARNING: No valid runtime data for epsilon = 10^-4 - skipping runtime plot")
    return nothing
  end
  
  min_runtime = minimum(all_runtimes)
  max_runtime = maximum(all_runtimes)
  
  # Calculate y-axis limits for log scale
  ylims_bottom = max(min_runtime * 0.8, 1e-20)
  ylims_top = max_runtime * 1.3
  
  # Generate ticks based on data range, excluding ticks >= 10^1
  data_min_exp = isfinite(log10(min_runtime)) ? floor(Int, log10(min_runtime)) : -3
  data_max_exp = isfinite(log10(max_runtime)) ? ceil(Int, log10(max_runtime)) : 0  # Cap at 10^0
  
  # Filter out 10^1 (exp = 1)
  yticks_pos_all = [10.0^exp for exp in data_min_exp:data_max_exp]
  yticks_labels_all = [exp == 0 ? latexstring("1") : latexstring("10^{$exp}") for exp in data_min_exp:data_max_exp]
  
  # Remove ticks >= 10^1, 10^-3, and 10^0 (1)
  valid_indices = [i for i in 1:length(yticks_pos_all) if yticks_pos_all[i] < 1e1 && abs(yticks_pos_all[i] - 1e-3) > 1e-10 && abs(yticks_pos_all[i] - 1.0) > 1e-10]
  yticks_pos = yticks_pos_all[valid_indices]
  yticks_labels = yticks_labels_all[valid_indices]
  
  # Get all unique sketching orders across all conditional orders
  all_sketching_orders = Int[]
  for (cond_order, df) in all_data
    sub_df = df[abs.(df.epsilon .- reference_epsilon) .< 1e-10, :]
    if !isempty(sub_df)
      append!(all_sketching_orders, unique(sub_df.order_recovery))
    end
  end
  order_recovery_vals = sort(unique(all_sketching_orders))
  xticks_pos = Float64.(order_recovery_vals)
  xticks_labels = [latexstring("$val") for val in order_recovery_vals]
  
  plt = plot(xlabel=latexstring("\\mathrm{Perturbative~Sketching~Order}~n_{\\mathrm{Sketch}}"),
             ylabel="",
             title=latexstring("\\mathrm{(d)~Runtime~(seconds)}"),
             legend=:bottomright,
             xticks=(xticks_pos, xticks_labels),
             yscale=:log10,
             yticks=(yticks_pos, yticks_labels),
             ylims=(ylims_bottom, ylims_top),
             fontsize=fontsize,
             tickfontsize=fontsize,
             guidefontsize=fontsize,
             legendfontsize=fontsize-2,
             titlefontsize=fontsize,
             bottom_margin=12Plots.mm,
             top_margin=14Plots.mm,
             linewidth=2.5,
             markersize=7,
             markerstrokewidth=0.1,
             dpi=300)
  
  # Plot runtime vs order_recovery for each conditional order (only for epsilon = 10^-4)
  conditional_orders = sort(collect(keys(all_data)))
  for (idx, cond_order) in enumerate(conditional_orders)
    df = all_data[cond_order]
    if !(:runtime in propertynames(df))
      continue
    end
    sub_df = df[abs.(df.epsilon .- reference_epsilon) .< 1e-10, :]
    if isempty(sub_df)
      continue
    end
    idx_sorted = sortperm(sub_df.order_recovery)
    x_vals = sub_df.order_recovery[idx_sorted]
    y_vals = sub_df.runtime[idx_sorted]
    
    # Plot line with legend showing only conditional order
    # (sketching order varies along x-axis, so we don't show it in the label)
    plot!(plt, x_vals, y_vals,
          marker=:o,
          label=latexstring("n_{\\mathrm{Cond}} = $(cond_order)"),
          color=idx,
          linewidth=2.5,
          markersize=7,
          markerstrokewidth=0.1)
  end
  
  return plt
end


if abspath(PROGRAM_FILE) == @__FILE__
  println("="^60)
  println("Running epsilon choice benchmark...")
  println("="^60)
  println("Configuration:")
  println("  Chain length (n_vertices): 8")
  println("  Order: 1")
  println("  Order recovery: [1, 2, 3, 4, 5, 6, 7]")
  println("  Sketching type: Perturbative")
  println("="^60)
  
  plt, all_data = run_epsilon_sweep(n_vertices=7, 
                                      order_recovery_vals=[1, 2, 4, 6],
                                      recompute=false)
  
  println("\n" * "="^60)
  println("Results summary:")
  println("="^60)
  for cond_order in [1, 2, 3]
    df = all_data[cond_order]
    println("\nConditional order = $cond_order:")
    for order_recovery in sort(unique(df.order_recovery))
      println("  Order recovery = $order_recovery:")
      sub_df = df[df.order_recovery .== order_recovery, :]
      for (idx, row) in enumerate(eachrow(sub_df))
        @printf("    Epsilon = %.6e, Max rel error = %.6e\n", row.epsilon, row.max_rel_error)
      end
    end
  end
  println("="^60)
end

