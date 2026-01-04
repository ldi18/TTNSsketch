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

function run_generative_regime_experiment(ttns, topology_name::String; 
                                         y_ticks_pos, y_ticks_labels, y_lims, 
                                         plot_label::String)
  d = length(ttns.x_indices)         # dimensions
  println("\n$topology_name: d = $d")
  set_warn_order(d+1)

  model = GraphicalModels.Ising_dGraphicalModel(ttns)
  probability_dict = Evaluate.probability_dict(model.ttns)

  sketching_kwargs = Dict{Symbol, Any}(
    :sketching_type => Sketching.Markov,
    :order => 1
  )

  # Drop probabilities to test: 0.1, 0.2, ..., 0.9
  drop_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

  # Store results
  unseen_mean_errors = Float64[]
  unseen_std_errors = Float64[]  # Standard deviation (sqrt of variance)
  percentages = Float64[]

  println("\n" * "="^60)
  println("Running generative regime experiment: $topology_name")
  println("="^60)

  for (idx, drop_prob) in enumerate(drop_probs)
    println("\nDrop probability: $drop_prob")
    
    # Randomly drop keys from training dict
    # Use different seed for each drop probability to ensure different selections
    Random.seed!(43 + idx)
    all_keys = collect(keys(probability_dict))
    n_total = length(all_keys)
    n_drop = Int(round(drop_prob * n_total))
    
    # Randomly select keys to drop (these will be not be seen during training)
    drop_indices = randperm(n_total)[1:n_drop]
    dropped_keys = Set(all_keys[drop_indices])
    seen_keys = Set(setdiff(all_keys, dropped_keys))
    println("We have $(length(all_keys)) keys in total, $(length(seen_keys)) seen, and $(length(dropped_keys)) dropped.")
    
    # Verify: dropped_keys and seen_keys are disjoint
    @assert isempty(intersect(dropped_keys, seen_keys)) "Error: dropped_keys and seen_keys overlap!"
    @assert length(dropped_keys) == n_drop "Error: wrong number of dropped keys!"
    @assert length(seen_keys) == n_total - n_drop "Error: wrong number of seen keys!"
    
    # Create training dict (only seen keys - model will NOT see dropped_keys)
    training_dict = Dict(key => probability_dict[key] for key in seen_keys)
    
    # Verify: training_dict does NOT contain any dropped keys
    @assert isempty(intersect(Set(keys(training_dict)), dropped_keys)) "Error: training_dict contains dropped keys!"
    
    # Train TTNS on seen keys only
    ttns_recov = deepcopy(ttns)
    CoreDeterminingEquations.compute_Gks!(training_dict, ttns_recov; sketching_kwargs, normalize_Gks=true)
    
    # Compute errors ONLY for unseen points (keys that were dropped and NOT in training)
    unseen_errors = Float64[]
    for key in dropped_keys
      # Double-check: this key should NOT be in training_dict
      @assert !haskey(training_dict, key) "Error: evaluating on a key that was in training!"
      
      p_ref = probability_dict[key]
      p_model = Evaluate.evaluate(ttns_recov, key)
      rel_error = p_ref == 0 ? (p_model == 0 ? 0.0 : Inf) : abs(p_ref - p_model) / p_ref
      push!(unseen_errors, rel_error)
    end
    
    unseen_mean = isempty(unseen_errors) ? NaN : mean(unseen_errors)
    unseen_std = isempty(unseen_errors) ? NaN : std(unseen_errors)
    
    percentage_seen = 1.0 - drop_prob
    
    push!(unseen_mean_errors, unseen_mean)
    push!(unseen_std_errors, unseen_std)
    push!(percentages, percentage_seen * 100)  # Convert to percentage
    
    println("  Percentage seen: $(@sprintf("%.1f", percentage_seen * 100))%")
    println("  Unseen points evaluated: $(length(unseen_errors))")
    println("  Mean rel error (unseen): $(isnan(unseen_mean) ? "N/A" : @sprintf("%.6e", unseen_mean))")
    println("  Std rel error (unseen): $(isnan(unseen_std) ? "N/A" : @sprintf("%.6e", unseen_std))")
    println("  Sum of TTNS: $(sum_ttns(ttns_recov))")
  end

  return unseen_mean_errors, unseen_std_errors, percentages
end

function create_plot(unseen_mean_errors, unseen_std_errors, percentages;
                     y_ticks_pos, y_ticks_labels, y_lims, plot_label::String,
                     plot_title::String,
                     plot_fontsize=18, legend_fontsize=16)
  unseen_mean_errors = [min(error, 1.0) for error in unseen_mean_errors]
                     # X-axis ticks
  x_ticks_pos = percentages
  x_ticks_labels = [latexstring(@sprintf("%.0f", x)) for x in x_ticks_pos]
  x_ticks = (x_ticks_pos, x_ticks_labels)

  # Y-axis ticks
  y_ticks = (y_ticks_pos, y_ticks_labels)

  # Sort by percentage for connected lines
  sort_idx = sortperm(percentages)

  # Create plot
  plt = plot(xlabel=latexstring("\\mathrm{Percentage~of~seen~points~(\\%)}"),
             ylabel=latexstring("\\mathrm{Mean~rel.~error}"),
             title=latexstring(plot_title),
             legend=:bottomleft,
             xticks=x_ticks,
             yticks=y_ticks,
             ylims=y_lims,
             fontsize=plot_fontsize,
             tickfontsize=plot_fontsize,
             guidefontsize=plot_fontsize,
             legendfontsize=legend_fontsize,
             titlefontsize=plot_fontsize,
             linewidth=2.5, markersize=7, markerstrokewidth=0.1)

  # Calculate error bar ranges using standard deviation
  unseen_lower_err = unseen_std_errors
  unseen_upper_err = unseen_std_errors

  # Plot unseen points with error bars
  plot!(plt, 
        [percentages[i] for i in sort_idx], 
        [unseen_mean_errors[i] for i in sort_idx],
        yerr=([unseen_lower_err[i] for i in sort_idx], [unseen_upper_err[i] for i in sort_idx]),
        marker=:o, label=latexstring(plot_label), 
        color=:blue, linewidth=2.5, markersize=7, markerstrokewidth=0.1,
        errorevery=1)

  return plt
end

# Run experiments for both topologies
ttns_binary = ExampleTopologies.BinaryTree(4)  # argument is depth, not vertex number!
ttns_linear = ExampleTopologies.Linear(15)  # Match number of vertices (2^4 - 1 = 15)

# Common plot settings
y_ticks_pos = [0, 0.2, 0.4, 0.6, 0.8, 1]
y_ticks_labels = [latexstring("0"), latexstring("0.2"), latexstring("0.4"), latexstring("0.6"), latexstring("0.8"), latexstring(">1")]
y_lims = (0, 1)
plot_fontsize = 18
legend_fontsize = plot_fontsize - 2

# Run experiment for binary tree
unseen_mean_errors_binary, unseen_std_errors_binary, percentages_binary = 
  run_generative_regime_experiment(ttns_binary, "BinaryTree(4)";
                                   y_ticks_pos=y_ticks_pos, 
                                   y_ticks_labels=y_ticks_labels,
                                   y_lims=y_lims,
                                   plot_label="\\mathrm{Binary~Tree}")

# Run experiment for linear topology
unseen_mean_errors_linear, unseen_std_errors_linear, percentages_linear = 
  run_generative_regime_experiment(ttns_linear, "Linear(15)";
                                   y_ticks_pos=y_ticks_pos, 
                                   y_ticks_labels=y_ticks_labels,
                                   y_lims=y_lims,
                                   plot_label="\\mathrm{Linear}")

# Create plots
plt_binary = create_plot(unseen_mean_errors_binary, unseen_std_errors_binary, percentages_binary;
                         y_ticks_pos=y_ticks_pos, 
                         y_ticks_labels=y_ticks_labels,
                         y_lims=y_lims,
                         plot_label="\\mathrm{Binary~Tree}",
                         plot_title="\\mathrm{Binary~tree~(15~vertices)}",
                         plot_fontsize=plot_fontsize,
                         legend_fontsize=legend_fontsize)

plt_linear = create_plot(unseen_mean_errors_linear, unseen_std_errors_linear, percentages_linear;
                         y_ticks_pos=y_ticks_pos, 
                         y_ticks_labels=y_ticks_labels,
                         y_lims=y_lims,
                         plot_label="\\mathrm{Linear}",
                         plot_title="\\mathrm{Linear~chain~(15~vertices)}",
                         plot_fontsize=plot_fontsize,
                         legend_fontsize=legend_fontsize)

# Combined plot side by side
fig_width = 1600
plt_combined = plot(plt_binary, plt_linear, layout=(1, 2), 
                    size=(fig_width, 600), dpi=300,
                    plot_title="",
                    left_margin=8Plots.mm, right_margin=8Plots.mm,
                    bottom_margin=12Plots.mm, top_margin=6Plots.mm)

savefig(plt_combined, joinpath(@__DIR__, "generative_regime_combined.pdf"))
println("\nSaved: generative_regime_combined.pdf")

println("\n" * "="^60)
println("Done!")
println("="^60)
