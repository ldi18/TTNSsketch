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

function run_generative_regime_experiment(ttns, topology_name::String; 
                                         y_ticks_pos, y_ticks_labels, y_lims, 
                                         plot_label::String,
                                         order::Int = 1,
                                         β::Real = 1.0)
  d = length(ttns.x_indices)         # dimensions
  println("\n$topology_name: d = $d, order = $order")
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

  # Number of samples to test: 10^2, 10^2.5, ..., 10^7
  sample_exponents = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
  n_samples_list = [Int(round(10.0^exp)) for exp in sample_exponents]

  # Store results
  mean_errors = Float64[]  # Mean relative error
  min_errors = Float64[]  # Minimum relative error
  max_errors = Float64[]  # Maximum relative error
  n_samples_used = Int[]
  unique_ratios = Float64[]  # Ratio of unique samples to 2^d

  println("\n" * "="^60)
  println("Running generative regime experiment (sample-based): $topology_name")
  println("="^60)

  for (idx, n_samples) in enumerate(n_samples_list)
    println("\nNumber of samples: $n_samples")
    
    # Generate samples from the true model
    Random.seed!(1234 + idx)  # Different seed for each sample size
    if order == 1
      sample_matrix = TTNSsketch.samples(model.ttns, n_samples; seed=1234 + idx)
    else
      # For order >= 2, sample directly from the probability_dict
      rng = Random.default_rng()
      sampled_keys = sample(rng, collect(keys(probability_dict)), ProbabilityWeights(collect(values(probability_dict))), n_samples)
      sample_matrix = reduce(vcat, [collect(key)' for key in sampled_keys])
    end
    
    # Train TTNS on samples
    ttns_recov = deepcopy(ttns)
    CoreDeterminingEquations.compute_Gks!(sample_matrix, ttns_recov; sketching_kwargs)
    
    # Compute errors on keys with significant probability
    # Filter out very small probabilities to avoid unreliable relative errors
    prob_threshold = 1e-10
    all_keys = collect(keys(probability_dict))
    keys_filtered = [key for key in all_keys if probability_dict[key] > prob_threshold]
    
    # Count unique samples that are in keys_filtered (above threshold)
    # Convert keys_filtered to a Set for fast lookup
    keys_filtered_set = Set(keys_filtered)
    unique_samples_above_threshold = Set{Tuple{Vararg{Int}}}()
    for i in 1:size(sample_matrix, 1)
      sample_tuple = Tuple(sample_matrix[i, :])
      if sample_tuple in keys_filtered_set
        push!(unique_samples_above_threshold, sample_tuple)
      end
    end
    n_unique_samples_above_threshold = length(unique_samples_above_threshold)
    
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
          push!(all_errors, 0.0)  # Both are zero
        end
        continue
      end
      
      # Compute relative error (p_ref > threshold, so safe to divide)
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
    
    # Calculate unique ratio: unique samples above threshold / total keys above threshold
    n_keys_above_threshold = length(keys_filtered)
    unique_ratio = n_keys_above_threshold > 0 ? n_unique_samples_above_threshold / n_keys_above_threshold : 0.0
    
    push!(mean_errors, mean_error)
    push!(min_errors, min_error)
    push!(max_errors, max_error)
    push!(n_samples_used, n_samples)
    push!(unique_ratios, unique_ratio)
    
    println("  Number of samples: $n_samples")
    println("  Unique samples above threshold: $n_unique_samples_above_threshold / $n_keys_above_threshold (out of $(2^d) total possible)")
    println("  Total keys: $(length(all_keys)), Keys above threshold ($prob_threshold): $(length(keys_filtered))")
    println("  Mean rel error (finite): $(isnan(mean_error) ? "N/A" : @sprintf("%.6e", mean_error))")
    println("  Min rel error (finite): $(isnan(min_error) ? "N/A" : @sprintf("%.6e", min_error))")
    println("  Max rel error (finite): $(isnan(max_error) ? "N/A" : @sprintf("%.6e", max_error))")
    println("  Sum of TTNS: $(sum_ttns(ttns_recov))")
  end

  # Print unique ratios as a list
  println("\n  Unique ratios (unique_samples_above_threshold / keys_above_threshold):")
  println("  [$(join([@sprintf("%.6f", r) for r in unique_ratios], ", "))]")

  return mean_errors, min_errors, max_errors, n_samples_used, unique_ratios
end

function create_error_plot(mean_errors_binary, min_errors_binary, max_errors_binary, n_samples_binary,
                           mean_errors_linear, min_errors_linear, max_errors_linear, n_samples_linear;
                           y_ticks_pos, y_ticks_labels, y_lims,
                           plot_fontsize=18, legend_fontsize=16)
  # X-axis ticks (use log scale for samples) - format like y ticks
  # Get unique sample values from both datasets
  all_samples = sort(unique(vcat(n_samples_binary, n_samples_linear)))
  x_ticks_pos = all_samples
  x_ticks_labels = map(x_ticks_pos) do x
    exp = log10(x)
    if exp == round(exp)
      # Integer exponent: 10^2, 10^3, etc.
      latexstring(@sprintf("10^{%.0f}", exp))
    else
      # Half-integer or other: 10^{2.5}, etc.
      latexstring(@sprintf("10^{%.1f}", exp))
    end
  end
  x_ticks = (x_ticks_pos, x_ticks_labels)
  
  # Determine y-axis range from actual data (don't cut off)
  finite_errors_binary = filter(e -> isfinite(e) && e > 0, mean_errors_binary)
  finite_errors_linear = filter(e -> isfinite(e) && e > 0, mean_errors_linear)
  all_finite_errors = vcat(finite_errors_binary, finite_errors_linear)
  
  if !isempty(all_finite_errors)
    y_min = 10.0^floor(log10(minimum(all_finite_errors)))
    y_max = 10.0^ceil(log10(maximum(all_finite_errors)))
    # Ensure we don't go below the minimum specified, but allow going above
    y_min = min(y_min, y_lims[1])
    y_max = max(y_max, y_lims[2])
    y_lims_actual = (y_min, y_max)
    
    # Generate y ticks to cover the full range
    y_min_exp = floor(log10(y_min))
    y_max_exp = ceil(log10(y_max))
    y_ticks_pos_actual = [10.0^exp for exp in y_min_exp:y_max_exp]
    y_ticks_labels_actual = [latexstring(@sprintf("10^{%.0f}", exp)) for exp in y_min_exp:y_max_exp]
    y_ticks = (y_ticks_pos_actual, y_ticks_labels_actual)
  else
    y_lims_actual = y_lims
    y_ticks = (y_ticks_pos, y_ticks_labels)
  end

  # Sort by number of samples for connected lines
  sort_idx_binary = sortperm(n_samples_binary)
  sort_idx_linear = sortperm(n_samples_linear)

  # Create plot (log-log)
  plt = plot(xlabel=latexstring("\\mathrm{Number~of~samples~}N"),
             ylabel=latexstring("\\mathrm{Mean~rel.~error}"),
             title=latexstring("\\mathrm{(a)~Mean~rel.~error~vs.~Number~of~samples~}N"),
             legend=:bottomleft,
             xscale=:log10,
             yscale=:log10,
             xticks=x_ticks,
             yticks=y_ticks,
             ylims=y_lims_actual,
             fontsize=plot_fontsize,
             tickfontsize=plot_fontsize,
             guidefontsize=plot_fontsize,
             legendfontsize=legend_fontsize,
             titlefontsize=plot_fontsize,
             linewidth=2.5, markersize=7, markerstrokewidth=0.1)

  # Calculate error bar ranges (distance below and above mean)
  error_below_binary = [max(0.0, mean_errors_binary[i] - min_errors_binary[i]) for i in 1:length(mean_errors_binary)]
  error_above_binary = [max(0.0, max_errors_binary[i] - mean_errors_binary[i]) for i in 1:length(mean_errors_binary)]
  error_below_linear = [max(0.0, mean_errors_linear[i] - min_errors_linear[i]) for i in 1:length(mean_errors_linear)]
  error_above_linear = [max(0.0, max_errors_linear[i] - mean_errors_linear[i]) for i in 1:length(mean_errors_linear)]

  # Plot both curves with error bars
  plot!(plt, 
        [n_samples_binary[i] for i in sort_idx_binary], 
        [mean_errors_binary[i] for i in sort_idx_binary],
        yerror=([error_below_binary[i] for i in sort_idx_binary], [error_above_binary[i] for i in sort_idx_binary]),
        marker=:o, label=latexstring("\\mathrm{Binary~Tree}"), 
        color=:blue, 
        linecolor=:blue, 
        markercolor=:blue,
        linewidth=2.5, markersize=7, markerstrokewidth=0.1,
        capsize=3, capthickness=1.5)
  
  plot!(plt, 
        [n_samples_linear[i] for i in sort_idx_linear], 
        [mean_errors_linear[i] for i in sort_idx_linear],
        yerror=([error_below_linear[i] for i in sort_idx_linear], [error_above_linear[i] for i in sort_idx_linear]),
        marker=:o, label=latexstring("\\mathrm{Linear}"), 
        color=:red,
        linecolor=:red,
        markercolor=:red,
        linewidth=2.5, markersize=7, markerstrokewidth=0.1,
        capsize=3, capthickness=1.5)

  return plt
end

function create_unique_ratio_plot(unique_ratios_binary, n_samples_binary, 
                                  unique_ratios_linear, n_samples_linear;
                                  plot_fontsize=18, legend_fontsize=16)
  # X-axis ticks (use log scale for samples)
  all_samples = sort(unique(vcat(n_samples_binary, n_samples_linear)))
  x_ticks_pos = all_samples
  x_ticks_labels = map(x_ticks_pos) do x
    exp = log10(x)
    if exp == round(exp)
      latexstring(@sprintf("10^{%.0f}", exp))
    else
      latexstring(@sprintf("10^{%.1f}", exp))
    end
  end
  x_ticks = (x_ticks_pos, x_ticks_labels)
  
  # Y-axis: log scale - determine range from data
  all_ratios = vcat(unique_ratios_binary, unique_ratios_linear)
  finite_ratios = filter(r -> isfinite(r) && r > 0, all_ratios)
  
  if !isempty(finite_ratios)
    y_min = 10.0^floor(log10(minimum(finite_ratios)))
    y_max = 10.0^ceil(log10(maximum(finite_ratios)))
    # Ensure y_max doesn't exceed 1.0
    y_max = min(y_max, 1.0)
    y_lims = (y_min, y_max)
    
    # Generate y ticks to cover the full range
    y_min_exp = floor(log10(y_min))
    y_max_exp = ceil(log10(y_max))
    y_ticks_pos = [10.0^exp for exp in y_min_exp:y_max_exp]
    y_ticks_labels = [latexstring(@sprintf("10^{%.0f}", exp)) for exp in y_min_exp:y_max_exp]
    y_ticks = (y_ticks_pos, y_ticks_labels)
  else
    y_lims = (0.01, 1.0)
    y_ticks_pos = [0.01, 0.1, 1.0]
    y_ticks_labels = [latexstring("10^{-2}"), latexstring("10^{-1}"), latexstring("10^{0}")]
    y_ticks = (y_ticks_pos, y_ticks_labels)
  end

  # Sort by number of samples for connected lines
  sort_idx_binary = sortperm(n_samples_binary)
  sort_idx_linear = sortperm(n_samples_linear)

  # Create plot (log-log scale)
  plt = plot(xlabel=latexstring("\\mathrm{Number~of~samples~}N"),
             ylabel=latexstring("\\mathrm{Fraction~of~seen~keys}"),
             title=latexstring("\\mathrm{(b)~Fraction~of~seen~keys~vs.~Number~of~samples~}N"),
             legend=:bottomright,
             xscale=:log10,
             yscale=:log10,
             xticks=x_ticks,
             yticks=y_ticks,
             ylims=y_lims,
             fontsize=plot_fontsize,
             tickfontsize=plot_fontsize,
             guidefontsize=plot_fontsize,
             legendfontsize=legend_fontsize,
             titlefontsize=plot_fontsize,
             linewidth=2.5, markersize=7, markerstrokewidth=0.1)

  # Plot both curves
  plot!(plt, 
        [n_samples_binary[i] for i in sort_idx_binary], 
        [unique_ratios_binary[i] for i in sort_idx_binary],
        marker=:o, label=latexstring("\\mathrm{Binary~Tree}"), 
        color=:blue, linewidth=2.5, markersize=7, markerstrokewidth=0.1)
  
  plot!(plt, 
        [n_samples_linear[i] for i in sort_idx_linear], 
        [unique_ratios_linear[i] for i in sort_idx_linear],
        marker=:o, label=latexstring("\\mathrm{Linear}"), 
        color=:red, linewidth=2.5, markersize=7, markerstrokewidth=0.1)

  return plt
end

# Run experiments for both topologies
ttns_binary = ExampleTopologies.BinaryTree(4)  # argument is depth, not vertex number!
ttns_linear = ExampleTopologies.Linear(15)  # Match number of vertices (2^4 - 1 = 15)

# Common plot settings (log-log plot)
y_ticks_pos = [0.001, 0.01, 0.1, 1.0]
y_ticks_labels = [latexstring("10^{-3}"), latexstring("10^{-2}"), latexstring("10^{-1}"), latexstring("10^{0}")]
y_lims = (0.001, 1.0)
plot_fontsize = 18
legend_fontsize = plot_fontsize - 2

# Run experiment for binary tree
mean_errors_binary, min_errors_binary, max_errors_binary, n_samples_binary, unique_ratios_binary = 
  run_generative_regime_experiment(ttns_binary, "BinaryTree(4)";
                                   y_ticks_pos=y_ticks_pos, 
                                   y_ticks_labels=y_ticks_labels,
                                   y_lims=y_lims,
                                   plot_label="\\mathrm{Binary~Tree}",
                                   order=1)

# Run experiment for linear topology
mean_errors_linear, min_errors_linear, max_errors_linear, n_samples_linear, unique_ratios_linear = 
  run_generative_regime_experiment(ttns_linear, "Linear(15)";
                                   y_ticks_pos=y_ticks_pos, 
                                   y_ticks_labels=y_ticks_labels,
                                   y_lims=y_lims,
                                   plot_label="\\mathrm{Linear}",
                                   order=1)

# Create plots
plt_error = create_error_plot(mean_errors_binary, min_errors_binary, max_errors_binary, n_samples_binary,
                              mean_errors_linear, min_errors_linear, max_errors_linear, n_samples_linear;
                              y_ticks_pos=y_ticks_pos, 
                              y_ticks_labels=y_ticks_labels,
                              y_lims=y_lims,
                              plot_fontsize=plot_fontsize,
                              legend_fontsize=legend_fontsize)

plt_unique = create_unique_ratio_plot(unique_ratios_binary, n_samples_binary,
                                     unique_ratios_linear, n_samples_linear;
                                     plot_fontsize=plot_fontsize,
                                     legend_fontsize=legend_fontsize)

# Combined plot side by side
fig_width = 1600
plt_combined = plot(plt_error, plt_unique, layout=(1, 2), 
                    size=(fig_width, 600), dpi=300,
                    plot_title="",
                    left_margin=8Plots.mm, right_margin=8Plots.mm,
                    bottom_margin=12Plots.mm, top_margin=6Plots.mm)

savefig(plt_combined, joinpath(@__DIR__, "generative_regime_samples_combined.pdf"))
println("\nSaved: generative_regime_samples_combined.pdf")

println("\n" * "="^60)
println("Done!")
println("="^60)
