using Graphs: SimpleGraph, SimpleDiGraph, is_tree, uniform_tree, binary_tree, reverse, edges
using ITensors: contract, onehot, scalar, set_warn_order
using ITensorNetworks: ITensorNetwork, siteinds, vertices, edges, add_vertex!, add_edge!
using NamedGraphs: NamedGraph, NamedDiGraph, NamedEdge, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_path_graph, named_grid
using Random
using LinearAlgebra: diagind
using Plots
using Printf
using Statistics
using LaTeXStrings
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "..", "ITensorNumericalAnalysis.jl", "src", "ITensorNumericalAnalysis.jl"))
using .ITensorNumericalAnalysis:
  indsnetwork,
  continuous_siteinds,
  interpolate,
  calculate_p,
  evaluate,
  IndsNetworkMap,
  complex_continuous_siteinds,
  vertex_digit,
  vertex_dimension,
  dig_and_dim

include(joinpath(@__DIR__, "..", "..", "TTNSsketch.jl", "src", "TTNSsketch.jl"))
using .TTNSsketch: CoreDeterminingEquations, TopologyDetection, TTNS, Sketching, SketchingSets
using .TTNSsketch: evaluate as evaluate_ttns
using .TTNSsketch.ErrorReporting: report_errors

function two_soliton(x, y)
  num = 12 * (3 + 4 * cosh(2 * x - 8 * y) + cosh(4 * x - 64 * y))
  den = 3 * cosh(x - 28 * y) + cosh(3 * x - 36 * y)
  den = den * den
  return -12 * num / den
end

"""
Returns a directed BTTN, with arrows pointing up to root.
"""
function bttn(dim, bits_per_fdim)
  g_directed = NamedDiGraph([(dig, d) for dig in 1:bits_per_fdim for d in 1:dim])
  g_undirected = NamedGraph([(dig, d) for dig in 1:bits_per_fdim for d in 1:dim])

  for i in 1:dim
    if i < dim
      add_edge!(g_directed, (1, i), (1, i+1))
      add_edge!(g_undirected, (1, i), (1, i+1))
    end
    if bits_per_fdim > 1
      add_edge!(g_directed, (2, i), (1, i))
      add_edge!(g_undirected, (2, i), (1, i))
      first_index_of_upper_level = 2
      first_index_of_lower_level = 3
      last_index_of_lower_level = 4
      while true
        for (j, v) in enumerate(first_index_of_lower_level:last_index_of_lower_level)
          if v <= bits_per_fdim
            add_edge!(g_undirected, (v, i), (first_index_of_upper_level + div(j - 1, 2), i))
            add_edge!(g_directed, (v, i), (first_index_of_upper_level + div(j - 1, 2), i))
          else
            @goto break_level_loop
          end
        end
        level_width = last_index_of_lower_level - first_index_of_lower_level + 1
        first_index_of_upper_level = first_index_of_lower_level
        first_index_of_lower_level = last_index_of_lower_level + 1
        last_index_of_lower_level = first_index_of_lower_level + 2 * level_width - 1
      end
      @label break_level_loop
    end
  end
  return (g_directed, g_undirected)
end

# Load the accessed points (They have format 0000000000000000 -72.0)
function load_accessed_points(filename, bits_per_fdim, fdim, n)
  accessed_points = Dict{Vector{Int}, Float64}()
  if !isfile(filename)
    return accessed_points
  end
  open(filename, "r") do io
    for line in eachline(io)
      parts = split(line, " ")
      if length(parts) < 2
        continue
      end
      bitstring_length = Int(length(parts[1]))
      point_INT = [parse(Int, parts[1][bitstring_length - (fdim - i + 1) * bits_per_fdim + 1: bitstring_length - (fdim - i) * bits_per_fdim], base=2) for i in 1:fdim]
      accessed_points[point_INT] = parse(Float64, parts[2])
    end
  end
  return accessed_points
end

# Convert discrete point tuple to continuous coordinates
function discrete_to_continuous(point_discrete, bits_per_fdim, fdim, n)
  x_bits = [point_discrete[j] - 1 for j in 1:bits_per_fdim]
  y_bits = [point_discrete[j + bits_per_fdim] - 1 for j in 1:bits_per_fdim]
  x_int = sum([x_bits[j] * (2^(bits_per_fdim - j)) for j in 1:bits_per_fdim])
  y_int = sum([y_bits[j] * (2^(bits_per_fdim - j)) for j in 1:bits_per_fdim])
  return [x_int / n, y_int / n]
end

# Convert continuous point to discrete tuple
function continuous_to_discrete(point_cont, bits_per_fdim, fdim, n)
  x_int = round(Int, point_cont[1] * n)
  y_int = round(Int, point_cont[2] * n)
  x_bits = [parse(Int, bit) + 1 for bit in bitstring(x_int)[length(bitstring(x_int))-(bits_per_fdim-1):length(bitstring(x_int))]]
  y_bits = [parse(Int, bit) + 1 for bit in bitstring(y_int)[length(bitstring(y_int))-(bits_per_fdim-1):length(bitstring(y_int))]]
  return Tuple(vcat([x_bits, y_bits]...))
end

# Compute relative errors for a set of points
function compute_relative_errors(points_discrete, evaluate_func, bits_per_fdim, fdim, n)
  rel_errors = Float64[]
  for point_discrete in points_discrete
    point_cont = discrete_to_continuous(point_discrete, bits_per_fdim, fdim, n)
    f_true = two_soliton(point_cont[1], point_cont[2])
    f_approx = evaluate_func(point_discrete)
    rel_err = abs(f_true) > 1e-10 ? abs(f_true - f_approx) / abs(f_true) : (abs(f_true - f_approx) > 1e-10 ? Inf : 0.0)
    push!(rel_errors, rel_err)
  end
  return rel_errors
end

Random.seed!(123)
bits_per_fdim = 5
fdim = 2
(g_directed, g_undirected) = bttn(fdim, bits_per_fdim)
s = continuous_siteinds(g_undirected; map_dimension=2)
n = 2^bits_per_fdim
f = x -> two_soliton(x[1], x[2])

# Generate all grid points
all_grid_points_discrete = Vector{NTuple{fdim * bits_per_fdim, Int64}}()
all_grid_points_cont = Vector{Vector{Float64}}()
for x in 0:n-1
  for y in 0:n-1
    point_discrete = Tuple(vcat([[parse(Int64, bit) + 1 for bit in bitstring(xi)[length(bitstring(xi))-(bits_per_fdim-1):length(bitstring(xi))]] for xi in [x, y]]...))
    push!(all_grid_points_discrete, point_discrete)
    push!(all_grid_points_cont, [x / n, y / n])
  end
end
total_points = length(all_grid_points_discrete)

# Storage for results (long format: each row is one result)
rows = Vector{NamedTuple}()

# CSV file path for saving/loading results
results_path = joinpath(@__DIR__, "compare_tci_ttns_sweeps_results.csv")

# Auto-reload if CSV exists (always use existing data if available)
csv_exists = isfile(results_path)
if csv_exists
  println("Loading existing results from: $results_path")
  df = CSV.read(results_path, DataFrame)
  # Explicitly filter out any perturbative data (shouldn't exist, but just in case)
  df = df[df.method .!= "TTNS_Perturbative", :]
  println("Loaded $(nrow(df)) rows from CSV (after filtering out perturbative data).")
else
  println("CSV file not found. Computing new results...")
  
  # Storage for results (long format: each row is one result)
  rows = Vector{NamedTuple}()
  
  # Loop over nsweeps
  for nsweeps in [1, 2, 3]
  println("\n" * "="^60)
  println("Running with nsweeps = $nsweeps")
  println("="^60)
  
  # Clear accessed points file before TCI run
  accessed_points_file_default = joinpath(@__DIR__, "accessed_points.txt")
  open(accessed_points_file_default, "w") do io
  end
  
  # Run TCI (this will write to accessed_points.txt)
  println("Running TCI interpolation...")
  tn = interpolate(f, s; nsweeps=nsweeps, maxdim=30, cutoff=1e-10, outputlevel=0, fdim=fdim, bits_per_fdim=bits_per_fdim)
  
  # Load accessed points for this nsweeps
  accessed_tci = load_accessed_points(accessed_points_file_default, bits_per_fdim, fdim, n)
  accessed_set_tci = Set([Tuple(p) for p in keys(accessed_tci)])
  n_accessed_tci = length(accessed_tci)
  percentage_accessed_tci = (n_accessed_tci / total_points) * 100
  
  println("  TCI accessed points: $n_accessed_tci ($(round(percentage_accessed_tci, digits=2))%)")
  
  # Separate TCI points into accessed and non-accessed
  tci_accessed_points_discrete = Vector{NTuple{fdim * bits_per_fdim, Int64}}()
  tci_non_accessed_points_discrete = Vector{NTuple{fdim * bits_per_fdim, Int64}}()
  
  for (i, point_discrete) in enumerate(all_grid_points_discrete)
    # Convert discrete point to INT format for lookup
    point_cont = all_grid_points_cont[i]
    point_int = [round(Int, point_cont[1] * n), round(Int, point_cont[2] * n)]
    
    if Tuple(point_int) in accessed_set_tci
      push!(tci_accessed_points_discrete, point_discrete)
    else
      push!(tci_non_accessed_points_discrete, point_discrete)
    end
  end
  
  # TCI evaluation function
  function evaluate_tci(point_discrete)
    point_cont = discrete_to_continuous(point_discrete, bits_per_fdim, fdim, n)
    return evaluate(tn, point_cont)
  end
  
  # Compute TCI relative errors
  println("  Computing TCI errors...")
  tci_rel_errors_acc = compute_relative_errors(tci_accessed_points_discrete, evaluate_tci, bits_per_fdim, fdim, n)
  tci_rel_errors_non_acc = compute_relative_errors(tci_non_accessed_points_discrete, evaluate_tci, bits_per_fdim, fdim, n)
  
  tci_mean_rel_error_accessed = isempty(tci_rel_errors_acc) ? NaN : maximum(tci_rel_errors_acc)
  tci_mean_rel_error_non_accessed = isempty(tci_rel_errors_non_acc) ? NaN : maximum(tci_rel_errors_non_acc)
  
  # Store TCI result
  push!(rows, (
    method = "TCI",
    nsweeps = nsweeps,
    order = missing,
    percentage = percentage_accessed_tci,
    rel_error_accessed = tci_mean_rel_error_accessed,
    rel_error_non_accessed = tci_mean_rel_error_non_accessed
  ))
  
  println("    TCI mean rel error (accessed): $(isnan(tci_mean_rel_error_accessed) ? "N/A" : @sprintf("%.6e", tci_mean_rel_error_accessed))")
  println("    TCI mean rel error (non-accessed): $(isnan(tci_mean_rel_error_non_accessed) ? "N/A" : @sprintf("%.6e", tci_mean_rel_error_non_accessed))")
  
  # TTNS part: select random points matching the number of accessed points from TCI
  println("\n  Setting up TTNS sketching with $n_accessed_tci random points...")
  
  # Select random points from all grid points
  Random.seed!(123 + nsweeps)  # Different seed for each nsweeps
  random_indices = randperm(total_points)[1:n_accessed_tci]
  ttns_accessed_points_discrete = all_grid_points_discrete[random_indices]
  ttns_accessed_set = Set(ttns_accessed_points_discrete)
  
  # Create function dictionary for TTNS (only accessed points)
  f_ttns = Dict{NTuple{fdim * bits_per_fdim, Int64}, Float64}()
  for point_discrete in ttns_accessed_points_discrete
    point_cont = discrete_to_continuous(point_discrete, bits_per_fdim, fdim, n)
    f_ttns[point_discrete] = two_soliton(point_cont[1], point_cont[2])
  end
  
  # Set up TTNS (shared for both methods)
  vertex_to_input_pos_map = Dict{Tuple{Int, Int}, Int}()
  for vertex in vertices(s.indsnetwork)
    (dig, dim) = dig_and_dim(first(s.indsnetwork.data_graph[vertex]))
    vertex_to_input_pos_map[vertex] = bits_per_fdim * (dim - 1) + dig
  end
  
  # Separate TTNS points into accessed and non-accessed
  ttns_non_accessed_points_discrete = Vector{NTuple{fdim * bits_per_fdim, Int64}}()
  for point_discrete in all_grid_points_discrete
    if point_discrete ∉ ttns_accessed_set
      push!(ttns_non_accessed_points_discrete, point_discrete)
    end
  end
  
  # ===== TTNS Markov Sketching =====
  # Use all sketching orders up to d-1 once
  # Get dimension d from a temporary TTNS
  ttns_temp = TTNS(g_directed; vertex_to_input_pos_map=vertex_to_input_pos_map)
  d = length(ttns_temp.x_indices)
  max_order = min(d - 1, 4)
  
  println("  Computing TTNS Gks (Markov) for orders 1 to $max_order...")
  
  for sketching_order in 1:max_order
    println("    Order $sketching_order/$max_order...")
    ttns_markov = TTNS(g_directed; vertex_to_input_pos_map=vertex_to_input_pos_map)
    sketching_kwargs_markov = Dict{Symbol, Any}(
      :sketching_type => Sketching.Markov,
      :order => sketching_order
    )
    set_warn_order(2*length(ttns_markov.x_indices))
    CoreDeterminingEquations.compute_Gks!(f_ttns, ttns_markov; sketching_kwargs=sketching_kwargs_markov)
    
    function evaluate_ttns_markov_wrapper(point_discrete)
      return evaluate_ttns(ttns_markov, point_discrete)
    end
    
    ttns_markov_rel_errors_acc = compute_relative_errors(ttns_accessed_points_discrete, evaluate_ttns_markov_wrapper, bits_per_fdim, fdim, n)
    ttns_markov_rel_errors_non_acc = compute_relative_errors(ttns_non_accessed_points_discrete, evaluate_ttns_markov_wrapper, bits_per_fdim, fdim, n)
    
    ttns_markov_mean_rel_error_accessed = isempty(ttns_markov_rel_errors_acc) ? NaN : maximum(ttns_markov_rel_errors_acc)
    ttns_markov_mean_rel_error_non_accessed = isempty(ttns_markov_rel_errors_non_acc) ? NaN : maximum(ttns_markov_rel_errors_non_acc)
    
    # Store TTNS Markov result
    push!(rows, (
      method = "TTNS_Markov",
      nsweeps = nsweeps,
      order = sketching_order,
      percentage = percentage_accessed_tci,
      rel_error_accessed = ttns_markov_mean_rel_error_accessed,
      rel_error_non_accessed = ttns_markov_mean_rel_error_non_accessed
    ))
    
    println("      TTNS (Markov, order=$sketching_order) max rel error (accessed): $(isnan(ttns_markov_mean_rel_error_accessed) ? "N/A" : @sprintf("%.6e", ttns_markov_mean_rel_error_accessed))")
    println("      TTNS (Markov, order=$sketching_order) max rel error (non-accessed): $(isnan(ttns_markov_mean_rel_error_non_accessed) ? "N/A" : @sprintf("%.6e", ttns_markov_mean_rel_error_non_accessed))")
  end
  end

  # Save results to CSV
  println("\nSaving results to: $results_path")
  df = DataFrame(rows)
  CSV.write(results_path, df)
  println("Results saved.")
end

# Extract data from DataFrame for plotting

# Separate TCI and TTNS Markov data (explicitly filter out any perturbative data)
tci_df = df[df.method .== "TCI", :]
ttns_markov_df = df[(df.method .== "TTNS_Markov") .& (.!ismissing.(df.order)), :]
ttns_markov_df = ttns_markov_df[ttns_markov_df.order .<= 4, :]

# Extract arrays for plotting
tci_percentages = tci_df.percentage
tci_rel_errors_accessed = tci_df.rel_error_accessed
tci_rel_errors_non_accessed = tci_df.rel_error_non_accessed

ttns_markov_percentages = ttns_markov_df.percentage
ttns_markov_orders = [Int(o) for o in ttns_markov_df.order]  # Convert to Int (no missing values after filtering)
ttns_markov_rel_errors_accessed = ttns_markov_df.rel_error_accessed
ttns_markov_rel_errors_non_accessed = ttns_markov_df.rel_error_non_accessed

# Plot styling to match epsilon_choice.pdf
plot_fontsize = 18
legend_fontsize = plot_fontsize - 2

function log10_ticks(values)
  finite_vals = [v for v in values if isfinite(v) && v > 0]
  if isempty(finite_vals)
    return (Float64[], Any[])
  end
  min_exp = floor(Int, log10(minimum(finite_vals)))
  max_exp = ceil(Int, log10(maximum(finite_vals)))
  tick_pos = [10.0^exp for exp in min_exp:max_exp]
  tick_labels = [latexstring("10^{$exp}") for exp in min_exp:max_exp]
  return (tick_pos, tick_labels)
end

# Tick labels as LaTeX strings
x_ticks_pos = sort(unique(tci_percentages))
x_ticks_labels = [latexstring(@sprintf("%.1f", x)) for x in x_ticks_pos]
x_ticks = (x_ticks_pos, x_ticks_labels)
all_rel_errors = vcat(tci_rel_errors_accessed,
                      tci_rel_errors_non_accessed,
                      ttns_markov_rel_errors_accessed,
                      ttns_markov_rel_errors_non_accessed)
y_ticks = log10_ticks(all_rel_errors)

# Create plots
println("\n" * "="^60)
println("Creating plots...")
println("="^60)

# Sort by percentage for connected lines
sort_idx_tci = sortperm(tci_percentages)

# For TTNS Markov, we need to group by order and sort within each order
# Get unique orders and percentages
unique_orders = sort(unique(ttns_markov_orders))
println("Found $(length(unique_orders)) unique TTNS Markov orders: $unique_orders")
ttns_markov_sorted_by_order = Dict{Int, Vector{Int}}()
for order in unique_orders
  order_indices = findall(x -> x == order, ttns_markov_orders)
  sort_idx_order = sortperm([ttns_markov_percentages[i] for i in order_indices])
  ttns_markov_sorted_by_order[order] = [order_indices[i] for i in sort_idx_order]
end

# Create individual plots for each order (showing both accessed and non-accessed)
plots_by_order = []
colors_markov = [:red, :orange, :purple, :brown, :pink, :gray, :olive, :cyan, :magenta, :yellow, :green, :blue, :darkred, :darkblue, :darkgreen, :darkorange, :darkviolet, :deeppink, :gold]

for (idx, order) in enumerate(unique_orders)
  order_indices = ttns_markov_sorted_by_order[order]
  color = idx <= length(colors_markov) ? colors_markov[idx] : :black
  marker = :o
  
  # Create plot for this order showing both accessed and non-accessed
  plt_order = plot(xlabel=latexstring("\\mathrm{Percentage~of~accessed~points~(\\%)}"),
                   ylabel=latexstring("\\mathrm{Max.~rel.~error}"),
                   title=latexstring("\\mathrm{Max.~rel.~error~vs.~frac.~accessed~points}"),
                   legend=:topright,
                   yscale=:log10,
                   xticks=x_ticks,
                   yticks=y_ticks,
                   fontsize=plot_fontsize,
                   tickfontsize=plot_fontsize,
                   guidefontsize=plot_fontsize,
                   legendfontsize=legend_fontsize,
                   titlefontsize=plot_fontsize,
                   linewidth=2, markersize=7, markerstrokewidth=0.1)
  
  # Add TCI line
  plot!(plt_order, tci_percentages[sort_idx_tci], tci_rel_errors_accessed[sort_idx_tci],
        marker=:o, label=latexstring("\\mathrm{TCI~(accessed)}"), color=:blue, linewidth=2, markersize=7, markerstrokewidth=0.1, linestyle=:solid)
  plot!(plt_order, tci_percentages[sort_idx_tci], tci_rel_errors_non_accessed[sort_idx_tci],
        marker=:o, label=latexstring("\\mathrm{TCI~(non\\,{-}\\,accessed)}"), color=:blue, linewidth=2, markersize=7, markerstrokewidth=0.1, linestyle=:dash)
  
  # Add TTNS Markov for this order
  plot!(plt_order, 
        [ttns_markov_percentages[i] for i in order_indices], 
        [ttns_markov_rel_errors_accessed[i] for i in order_indices],
        marker=marker, label=latexstring("\\mathrm{TTNS~(accessed)}"), 
        color=color, linewidth=2, markersize=7, markerstrokewidth=0.1, linestyle=:solid)
  plot!(plt_order, 
        [ttns_markov_percentages[i] for i in order_indices], 
        [ttns_markov_rel_errors_non_accessed[i] for i in order_indices],
        marker=marker, label=latexstring("\\mathrm{TTNS~(non\\,{-}\\,accessed)}"), 
        color=color, linewidth=2, markersize=7, markerstrokewidth=0.1, linestyle=:dash)
  
  push!(plots_by_order, plt_order)
end

# Keep figure width consistent with epsilon_choice.pdf
fig_width = 1600

# Also create the combined plot with all orders on one plot (for comparison)
accessed_exps = [exp for exp in -15:2 if isodd(exp)]
accessed_y_ticks = ([10.0^exp for exp in accessed_exps],
                    [latexstring("10^{$exp}") for exp in accessed_exps])

plt_accessed = plot(xlabel=latexstring("\\mathrm{Percentage~of~accessed~points~(\\%)}"),
                   ylabel=latexstring("\\mathrm{Max.~rel.~error}"),
                   title=latexstring("\\mathrm{Seen~points:~Max.~rel.~error~vs.~frac.~accessed~points}"),
                   legend=:left,
                   yscale=:log10,
                   xticks=x_ticks,
                   yticks=accessed_y_ticks,
                   ylims=(5e-16, 1e2),
                   fontsize=plot_fontsize,
                   tickfontsize=plot_fontsize,
                   guidefontsize=plot_fontsize,
                   legendfontsize=legend_fontsize,
                   titlefontsize=plot_fontsize,
                   linewidth=2.5, markersize=7, markerstrokewidth=0.1)

plot!(plt_accessed, tci_percentages[sort_idx_tci], tci_rel_errors_accessed[sort_idx_tci],
      marker=:o, label=latexstring("\\mathrm{TCI}"), color=:blue, linewidth=2.5, markersize=7, markerstrokewidth=0.1)

for (idx, order) in enumerate(unique_orders)
  order_indices = ttns_markov_sorted_by_order[order]
  color = idx <= length(colors_markov) ? colors_markov[idx] : :black
  marker = :o
  plot!(plt_accessed, 
        [ttns_markov_percentages[i] for i in order_indices], 
        [ttns_markov_rel_errors_accessed[i] for i in order_indices],
        marker=marker, label=latexstring("\\mathrm{TTNS~(Markov,~order=$order)}"), 
        color=color, linewidth=2.5, markersize=7, markerstrokewidth=0.1)
end

plt_non_accessed = plot(xlabel=latexstring("\\mathrm{Percentage~of~accessed~points~(\\%)}"),
                        ylabel=latexstring("\\mathrm{Max.~rel.~error}"),
                        title=latexstring("\\mathrm{Unseen~points:~Max.~rel.~error~vs.~frac.~accessed}"),
                        legend=false,
                        yscale=:log10,
                        xticks=x_ticks,
                        yticks=accessed_y_ticks,
                        ylims=(5e-16, 1e2),
                        fontsize=plot_fontsize,
                        tickfontsize=plot_fontsize,
                        guidefontsize=plot_fontsize,
                        legendfontsize=legend_fontsize,
                        titlefontsize=plot_fontsize,
                        linewidth=2.5, markersize=7, markerstrokewidth=0.1)

plot!(plt_non_accessed, tci_percentages[sort_idx_tci], tci_rel_errors_non_accessed[sort_idx_tci],
      marker=:o, label=latexstring("\\mathrm{TCI}"), color=:blue, linewidth=2.5, markersize=7, markerstrokewidth=0.1)

for (idx, order) in enumerate(unique_orders)
  order_indices = ttns_markov_sorted_by_order[order]
  color = idx <= length(colors_markov) ? colors_markov[idx] : :black
  marker = :o
  plot!(plt_non_accessed, 
        [ttns_markov_percentages[i] for i in order_indices], 
        [ttns_markov_rel_errors_non_accessed[i] for i in order_indices],
        marker=marker, label=latexstring("\\mathrm{TTNS~(Markov,~order=$order)}"), 
        color=color, linewidth=2.5, markersize=7, markerstrokewidth=0.1)
end

# Combined plot side by side (all orders on one plot)
plt_combined = plot(plt_accessed, plt_non_accessed, layout=(1, 2), 
                    size=(fig_width, 600), dpi=300,
                    plot_title="",
                    left_margin=8Plots.mm, right_margin=8Plots.mm,
                    bottom_margin=12Plots.mm, top_margin=6Plots.mm)
savefig(plt_combined, joinpath(@__DIR__, "compare_tci_ttns_combined.pdf"))
println("Saved: compare_tci_ttns_combined.pdf")

println("\nPlot values table:")
plot_values = vcat(
  DataFrame(method=fill("TCI", length(tci_percentages)),
            order=fill(missing, length(tci_percentages)),
            percentage=tci_percentages,
            rel_error_accessed=tci_rel_errors_accessed,
            rel_error_non_accessed=tci_rel_errors_non_accessed),
  DataFrame(method=fill("TTNS_Markov", length(ttns_markov_percentages)),
            order=ttns_markov_orders,
            percentage=ttns_markov_percentages,
            rel_error_accessed=ttns_markov_rel_errors_accessed,
            rel_error_non_accessed=ttns_markov_rel_errors_non_accessed)
)
sort!(plot_values, [:method, :order, :percentage])
show(plot_values, allrows=true, allcols=true)
println()

println("\n" * "="^60)
println("Done!")
println("="^60)
