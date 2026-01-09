include(joinpath(@__DIR__, "..", "..", "TTNSsketch.jl", "src", "TTNSsketch.jl"))
using .TTNSsketch
using .TTNSsketch.ErrorReporting: report_errors
using .TTNSsketch.TopologyNotation: vertices, C
using .TTNSsketch.Sketching: compute_Zwk
using .TTNSsketch.Evaluate: sum_ttns
using Printf
using ITensors: set_warn_order, inds, tags, array
using LinearAlgebra
using Plots
gr()  # Use GR backend for LaTeX support
using LaTeXStrings
using Random
using Statistics
using StatsBase: sample, ProbabilityWeights
using BenchmarkTools
using CSV
using DataFrames

function run_linear_scaling_experiment(; n_samples::Int = 10^7,
                                       d_range::Vector{Int} = collect(1:16),
                                       order::Int = 1,
                                       β::Real = 1.0,
                                       seed::Int = 1234)
  # Fixed number of samples
  println("\n" * "="^60)
  println("Running linear scaling experiment")
  println("Number of samples: $n_samples")
  println("Topology: Linear chain")
  println("Varying d from $(minimum(d_range)) to $(maximum(d_range))")
  println("="^60)

  # Store results
  mean_errors = Float64[]
  min_errors = Float64[]
  max_errors = Float64[]
  runtimes = Float64[]
  d_values = Int[]

  for d in d_range
    println("\n" * "-"^60)
    println("d = $d")
    println("-"^60)
    
    # Create linear topology with d vertices
    ttns = ExampleTopologies.Linear(d)
    set_warn_order(d+1)

    # Create probability dictionary based on order
    if order == 1
      model = GraphicalModels.Ising_dGraphicalModel(ttns)
      probability_dict = Evaluate.probability_dict(model.ttns)
    else
      # Use higher_order_probability_dict for order >= 2
      probability_dict = GraphicalModels.higher_order_probability_dict(ttns; β=β, order=order)
    end

    sketching_kwargs = Dict{Symbol, Any}(
      :sketching_type => Sketching.Markov,
      :order => order
    )

    # Generate samples from the true model
    Random.seed!(seed)
    if order == 1
      sample_matrix = TTNSsketch.samples(model.ttns, n_samples; seed=seed)
    else
      # For order >= 2, sample directly from the probability_dict
      rng = Random.default_rng()
      sampled_keys = sample(rng, collect(keys(probability_dict)), ProbabilityWeights(collect(values(probability_dict))), n_samples)
      sample_matrix = reduce(vcat, [collect(key)' for key in sampled_keys])
    end
    
    # Train TTNS on samples and measure runtime
    # Benchmark the compute_Gks! function (it modifies in place, so we need to create fresh copy inside)
    bench_result = @benchmark begin
      ttns_recov_bench = deepcopy($ttns)
      CoreDeterminingEquations.compute_Gks!(
        $sample_matrix, ttns_recov_bench; 
        sketching_kwargs=$sketching_kwargs
      )
    end evals=1
    
    runtime_seconds = mean(bench_result.times) / 1e9  # Convert nanoseconds to seconds
    
    # Now run once more to get the actual recovered TTNS for error computation
    ttns_recov = deepcopy(ttns)
    CoreDeterminingEquations.compute_Gks!(sample_matrix, ttns_recov; sketching_kwargs)
    
    # Compute errors on keys with significant probability
    prob_threshold = 1e-10
    all_keys = collect(keys(probability_dict))
    keys_filtered = [key for key in all_keys if probability_dict[key] > prob_threshold]
    
    all_errors = Float64[]
    invalid_evaluations = 0
    
    for key in keys_filtered
      p_ref = probability_dict[key]
      p_model = Evaluate.evaluate(ttns_recov, key)
      
      # Check for invalid model values
      if !isfinite(p_model) || isnan(p_model) || isinf(p_model)
        invalid_evaluations += 1
        if p_ref > prob_threshold
          push!(all_errors, Inf)
        else
          push!(all_errors, 0.0)
        end
        continue
      end
      
      # Compute relative error
      rel_error = abs(p_ref - p_model) / p_ref
      push!(all_errors, rel_error)
    end
    
    if invalid_evaluations > 0
      println("  WARNING: $invalid_evaluations invalid model evaluations (NaN/Inf) out of $(length(keys_filtered))")
    end
    
    # Filter out Inf values for statistics
    finite_errors = filter(e -> isfinite(e) && !isinf(e), all_errors)
    
    mean_error = isempty(finite_errors) ? NaN : mean(finite_errors)
    min_error = isempty(finite_errors) ? NaN : minimum(finite_errors)
    max_error = isempty(finite_errors) ? NaN : maximum(finite_errors)
    
    push!(mean_errors, mean_error)
    push!(min_errors, min_error)
    push!(max_errors, max_error)
    push!(runtimes, runtime_seconds)
    push!(d_values, d)
    
    println("  Total keys: $(length(all_keys)), Keys above threshold ($prob_threshold): $(length(keys_filtered))")
    println("  Mean rel error (finite): $(isnan(mean_error) ? "N/A" : @sprintf("%.6e", mean_error))")
    println("  Min rel error (finite): $(isnan(min_error) ? "N/A" : @sprintf("%.6e", min_error))")
    println("  Max rel error (finite): $(isnan(max_error) ? "N/A" : @sprintf("%.6e", max_error))")
    println("  Runtime: $(@sprintf("%.4f", runtime_seconds)) seconds")
  end

  return mean_errors, min_errors, max_errors, runtimes, d_values
end

function create_plots(mean_errors, min_errors, max_errors, runtimes, d_values;
                     plot_fontsize=18, legend_fontsize=16)
  
  # Determine y-axis range for error plot (linear scale)
  # Set upper limit to 5 * 10^-2 (0.05)
  y_lims_error = (0.0, 0.05)
  
  # Generate y-ticks: 0, 1·10^-2, 2·10^-2, ..., up to 5·10^-2
  max_tick = 5  # 5 * 10^-2 = 0.05
  y_ticks_pos_error = [i * 1e-2 for i in 0:max_tick]  # 0, 0.01, 0.02, 0.03, 0.04, 0.05
  y_ticks_labels_error = [i == 0 ? latexstring("0") : latexstring(@sprintf("%d \\cdot 10^{-2}", i)) for i in 0:max_tick]
  y_ticks_error = (y_ticks_pos_error, y_ticks_labels_error)

  # Determine y-axis range for runtime plot (linear scale)
  y_min_runtime = minimum(runtimes)
  y_max_runtime = maximum(runtimes)
  # Add some padding
  y_range_runtime = y_max_runtime - y_min_runtime
  y_min_runtime = max(0.0, y_min_runtime - 0.1 * y_range_runtime)
  y_max_runtime = y_max_runtime + 0.1 * y_range_runtime
  y_lims_runtime = (y_min_runtime, y_max_runtime)
  
  # Generate y-ticks for runtime plot: even integers 0, 2, 4, 6, 8, 10 (only up to 10)
  y_ticks_pos_runtime = [0, 2, 4, 6, 8, 10]
  y_ticks_labels_runtime = [latexstring(@sprintf("%d", tick)) for tick in y_ticks_pos_runtime]
  y_ticks_runtime = (y_ticks_pos_runtime, y_ticks_labels_runtime)

  # X-axis ticks (d values) - linear scale
  x_ticks_pos = d_values
  x_ticks_labels = [latexstring(@sprintf("%d", d)) for d in d_values]
  x_ticks = (x_ticks_pos, x_ticks_labels)

  # Create error plot (left) - linear-linear
  plt_error = plot(xlabel=latexstring("\\mathrm{System~size~}d"),
                   ylabel=latexstring("\\mathrm{Mean~rel.~error}"),
                   title=latexstring("\\mathrm{(a)~Mean~rel.~error~vs.~System~size~}d"),
                   legend=false,
                   xscale=:identity,
                   yscale=:identity,
                   xticks=x_ticks,
                   yticks=y_ticks_error,
                   ylims=y_lims_error,
                   fontsize=plot_fontsize,
                   tickfontsize=plot_fontsize,
                   guidefontsize=plot_fontsize,
                   legendfontsize=legend_fontsize,
                   titlefontsize=plot_fontsize,
                   linewidth=2.5, markersize=7, markerstrokewidth=0.1)

  # Calculate error bar ranges (distance below and above mean)
  error_below = [max(0.0, mean_errors[i] - min_errors[i]) for i in 1:length(mean_errors)]
  error_above = [max(0.0, max_errors[i] - mean_errors[i]) for i in 1:length(mean_errors)]
  
  plot!(plt_error, 
        d_values, 
        mean_errors,
        yerror=(error_below, error_above),
        marker=:o, 
        color=:blue, 
        errorbarcolor=:blue,
        linewidth=2.5, markersize=7, markerstrokewidth=0.1)

  # Create runtime plot (right) - linear-linear
  plt_runtime = plot(xlabel=latexstring("\\mathrm{System~size~}d"),
                     ylabel=latexstring("\\mathrm{Runtime~(second)}"),
                     title=latexstring("\\mathrm{(b)~Runtime~vs.~System~size~}d"),
                     legend=false,
                     xscale=:identity,
                     yscale=:identity,
                     xticks=x_ticks,
                     yticks=y_ticks_runtime,
                     ylims=y_lims_runtime,
                   fontsize=plot_fontsize,
                   tickfontsize=plot_fontsize,
                   guidefontsize=plot_fontsize,
                   legendfontsize=legend_fontsize,
                   titlefontsize=plot_fontsize,
                   linewidth=2.5, markersize=7, markerstrokewidth=0.1)

  plot!(plt_runtime, 
        d_values, 
        runtimes,
        marker=:o, 
        color=:blue, linewidth=2.5, markersize=7, markerstrokewidth=0.1)

  return plt_error, plt_runtime
end

# Run experiment
d_range = 1:16
n_samples = 10^7
order = 1
β = 1.0
seed = 1234

# Check if CSV file exists and load data
csv_path = joinpath(@__DIR__, "linear_scaling_results.csv")
if isfile(csv_path)
  println("Loading data from CSV: $csv_path")
  df = CSV.read(csv_path, DataFrame)
  d_values = collect(df.d)
  mean_errors = collect(df.mean_error)
  # Handle backward compatibility: if std_error column exists, estimate min/max from it
  if :min_error in propertynames(df) && :max_error in propertynames(df)
    min_errors = collect(df.min_error)
    max_errors = collect(df.max_error)
  elseif :std_error in propertynames(df)
    # Estimate min/max from std (assuming normal distribution: min ≈ mean - 2*std, max ≈ mean + 2*std)
    std_errors = collect(df.std_error)
    min_errors = [max(0.0, mean_errors[i] - 2.0 * std_errors[i]) for i in 1:length(mean_errors)]
    max_errors = [mean_errors[i] + 2.0 * std_errors[i] for i in 1:length(mean_errors)]
    println("WARNING: CSV missing min_error/max_error columns. Estimated from std_error.")
  else
    error("CSV missing required error columns. Please recompute.")
  end
  runtimes = collect(df.runtime)
  println("Loaded $(length(d_values)) data points")
else
  println("Running experiment (CSV not found)")
  mean_errors, min_errors, max_errors, runtimes, d_values = 
    run_linear_scaling_experiment(; n_samples=n_samples, d_range=collect(d_range), order=order, β=β, seed=seed)
  
  # Save results to CSV
  df = DataFrame(
    d = d_values,
    mean_error = mean_errors,
    min_error = min_errors,
    max_error = max_errors,
    runtime = runtimes
  )
  CSV.write(csv_path, df)
  println("\nSaved results to CSV: $csv_path")
end

# Create plots
plot_fontsize = 18
legend_fontsize = plot_fontsize - 2

plt_error, plt_runtime = create_plots(mean_errors, min_errors, max_errors, runtimes, d_values;
                                      plot_fontsize=plot_fontsize,
                                      legend_fontsize=legend_fontsize)

# Combined plot side by side
fig_width = 1600
plt_combined = plot(plt_error, plt_runtime, layout=(1, 2), 
                    size=(fig_width, 600), dpi=300,
                    plot_title="",
                    left_margin=8Plots.mm, right_margin=8Plots.mm,
                    bottom_margin=12Plots.mm, top_margin=6Plots.mm)

savefig(plt_combined, joinpath(@__DIR__, "linear_scaling_combined.pdf"))
println("\nSaved: linear_scaling_combined.pdf")

println("\n" * "="^60)
println("Done!")
println("="^60)

