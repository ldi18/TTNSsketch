using CSV
using DataFrames
using Random
using Graphs: nv, ne, edges, src, dst, SimpleDiGraph, add_edge!, vertices
using GraphRecipes
using NetworkLayout
using Plots
using LaTeXStrings

include(joinpath(@__DIR__, "..", "..", "TTNSsketch.jl", "src", "TTNSsketch.jl"))
using .TTNSsketch.TopologyDetection: maximum_spanning_tree_recovery
using .TTNSsketch.CoreDeterminingEquations
using .TTNSsketch.Evaluate: evaluate
using .TTNSsketch.Sketching
using .TTNSsketch.Structs
using .TTNSsketch.TopologyNotation: set_root!

# Load and encode data
const DATA_PATH = joinpath(@__DIR__, "online_shoppers_intention.csv")
df = CSV.read(DATA_PATH, DataFrame)

# Use all columns
col_labels = string.(names(df))
n_cols = length(col_labels)
println("Using all $n_cols columns: $(join(col_labels, ", "))")

const REPLACEMENT_RULES = Dict(
  "Month" => Dict("Jan"=>1, "Feb"=>2, "Mar"=>3, "Apr"=>4, "May"=>5, "June"=>6,
                  "Jul"=>7, "Aug"=>8, "Sep"=>9, "Oct"=>10, "Nov"=>11, "Dec"=>12),
  "VisitorType" => Dict("Returning_Visitor"=>1, "New_Visitor"=>2, "Other"=>3),
  "Weekend" => Dict("FALSE"=>1, "TRUE"=>2),
  "Revenue" => Dict("FALSE"=>1, "TRUE"=>2)
)

function encode(value, col_name::AbstractString)
  rules = get(REPLACEMENT_RULES, col_name, nothing)
  rules === nothing && return value
  ismissing(value) && return missing
  key = value isa Bool ? (value ? "TRUE" : "FALSE") : string(value)
  return get(rules, key, value)
end

# Discretize columns
col_values = Vector{Vector}(undef, n_cols)
discrete_offsets = zeros(Int, n_cols)
for (j, col_name) in enumerate(names(df))
  col_vals = df[!, col_name]
  encoded_vals = encode.(col_vals, Ref(col_labels[j]))
  
  if haskey(REPLACEMENT_RULES, col_labels[j])
    col_values[j] = Int.(encoded_vals)
    discrete_offsets[j] = minimum(col_values[j]) == 0 ? 1 : 0
  else
    # Numeric columns: bin into 2 bins
    numeric_vals = Float64.(coalesce.(encoded_vals, NaN))
    valid_vals = filter(isfinite, numeric_vals)
    vmin, vmax = extrema(valid_vals)
    edges = vmin == vmax ? [vmin-0.5, vmax+0.5] : collect(range(vmin, vmax; length=3))
    col_values[j] = [clamp(searchsortedlast(edges, Float64(v)), 1, 2) for v in numeric_vals]
    discrete_offsets[j] = 0
  end
end

# Step 1: Compute probability dict from full dataset
println("Step 1: Computing probability dict from full dataset...")
function build_joint_dict(values_by_col)
  counts = Dict{NTuple{length(values_by_col), Float64}, Int}()
  for row in 1:length(values_by_col[1])
    key = ntuple(j -> Float64(values_by_col[j][row]), length(values_by_col))
    counts[key] = get(counts, key, 0) + 1
  end
  total = sum(values(counts))
  return Dict(k => v / total for (k, v) in counts)
end

probability_dict = build_joint_dict(col_values)

# Calculate total possible patterns in input space (product of all input dimensions)
# First, determine the dimensions for each column
col_dims = [maximum(col_values[i]) + discrete_offsets[i] for i in 1:n_cols]
total_possible_patterns = prod(col_dims)

num_unique_samples = length(probability_dict)
println("  Number of unique (distinct) samples in dataset: $num_unique_samples")
println("  Total possible patterns in input space: $total_possible_patterns (product of dimensions: $(join(col_dims, " × ")) = $total_possible_patterns)")
println("  Total data points: $(nrow(df))")

# Step 2: Use probability dict to determine Chow-Liu tree
println("\nStep 2: Determining Chow-Liu tree from probability dict...")
revenue_index = findfirst(==("Revenue"), col_labels)
if revenue_index === nothing
  error("Revenue column not found")
end

tree = maximum_spanning_tree_recovery(probability_dict; max_degree=3, root_vertex=revenue_index, bmi_threshold=1e-3)
println("  Tree determined with $(nv(tree)) vertices and $(ne(tree)) edges")

# Visualize the Chow-Liu tree
function visualize_chow_liu_tree(tree, col_labels, revenue_index)
  # Set root to revenue_index to ensure proper tree structure
  rooted_tree = deepcopy(tree)
  set_root!(rooted_tree, revenue_index)
  
  # Find all reachable vertices from the root (in case of disconnected components)
  reachable_vertices = Set{Int}([revenue_index])
  function collect_reachable(v)
    for e in edges(rooted_tree)
      if src(e) == v
        if !(dst(e) in reachable_vertices)
          push!(reachable_vertices, dst(e))
          collect_reachable(dst(e))
        end
      end
    end
  end
  collect_reachable(revenue_index)
  
  # Filter to only reachable vertices
  reachable_list = sort(collect(reachable_vertices))
  
  # Check if we have any edges in the reachable component
  reachable_edges = [e for e in edges(rooted_tree) if src(e) in reachable_vertices && dst(e) in reachable_vertices]
  
  if isempty(reachable_edges)
    println("  Warning: Revenue vertex is isolated (no edges). Using spring layout for all vertices.")
    # If root is isolated, just show all vertices with a different layout
    g = SimpleDiGraph(nv(rooted_tree))
    for e in edges(rooted_tree)
      add_edge!(g, dst(e), src(e))
    end
    node_labels = [col_labels[i] for i in 1:nv(rooted_tree)]
    node_colors = [i == revenue_index ? :red : :lightblue for i in 1:nv(rooted_tree)]
    
    plt = plot(g; 
               names=node_labels,
               curves=false, 
               nodeshape=:circle, 
               title="Chow-Liu Tree (Forest - Revenue isolated)",
               fontsize=8, 
               nodesize=0.08,
               nodecolor=node_colors,
               titlefontsize=14,
               aspect_ratio=1,
               linewidth=2,
               method=:spring,
               arrow=false,
               size=(1400, 1000))
  else
    # Create a mapping from original vertex indices to new indices
    vertex_map = Dict(v => i for (i, v) in enumerate(reachable_list))
    
    # Convert NamedDiGraph to SimpleDiGraph for plotting (only reachable vertices)
    g = SimpleDiGraph(length(reachable_list))
    for e in reachable_edges
      # Reverse: parent -> child for visualization
      add_edge!(g, vertex_map[dst(e)], vertex_map[src(e)])
    end
    
    # Use column labels as node names (only for reachable vertices)
    node_labels = [col_labels[reachable_list[i]] for i in 1:length(reachable_list)]
    
    # Highlight the root (Revenue)
    root_pos = vertex_map[revenue_index]
    node_colors = [i == root_pos ? :red : :lightblue for i in 1:length(reachable_list)]
    
    plt = plot(g; 
               names=node_labels,
               curves=false, 
               nodeshape=:circle, 
               title="Chow-Liu Tree (Connected Component with Revenue)",
               fontsize=9, 
               nodesize=0.1,
               nodecolor=node_colors,
               titlefontsize=14,
               aspect_ratio=1,
               linewidth=2,
               method=:buchheim,
               arrow=false,
               size=(1200, 900))
  end
  
  output_pdf = joinpath(@__DIR__, "chow_liu_tree.pdf")
  savefig(plt, output_pdf)
  println("  Saved tree visualization to: $output_pdf")
  println("  Note: Tree has $(nv(tree)) vertices and $(ne(tree)) edges (may be disconnected)")
  println("  Visualized component contains $(length(reachable_list)) vertices including Revenue")
  return plt
end

tree_plot = visualize_chow_liu_tree(tree, col_labels, revenue_index)

# Step 3: Build sample matrix from all data
println("\nStep 3: Building sample matrix...")
sample_matrix = hcat([col_values[j] .+ discrete_offsets[j] for j in 1:n_cols]...)
println("  Sample matrix shape: $(size(sample_matrix))")

vertex_input_dim = Dict(i => maximum(col_values[i]) + discrete_offsets[i] for i in 1:n_cols)

function build_input_from_key(key)
  input_vals = Vector{Float64}(undef, n_cols)
  for j in 1:n_cols
    input_vals[j] = Float64(key[j]) + discrete_offsets[j]
  end
  return Tuple(Int64.(input_vals))
end

# Function to train TTNS and compute KL divergence for a given order
function train_and_evaluate(order::Int)
  println("\n" * "="^80)
  println("Training TTNS with order=$order...")
  println("="^80)
  
  # Train TTNS
  ttns = Structs.TTNS(tree; vertex_to_input_pos_map=Dict(i => i for i in 1:n_cols), vertex_input_dim=vertex_input_dim)
  # Ensure Revenue is set as the root node
  set_root!(ttns.tree, revenue_index)
  CoreDeterminingEquations.compute_Gks!(sample_matrix, ttns; 
    sketching_kwargs=Dict(:sketching_type => Sketching.Markov, :order => order))
  println("  TTNS training complete")
  
  # Normalize TTNS distribution
  println("\n  Computing KL divergence...")
  ttns_probs = Dict{NTuple{n_cols, Float64}, Float64}()
  let ttns_sum = 0.0
    for key in keys(probability_dict)
      input_tuple = build_input_from_key(key)
      prob = abs(evaluate(ttns, input_tuple))
      ttns_probs[key] = prob
      ttns_sum += prob
    end
    
    # Normalize TTNS probabilities
    if ttns_sum > 0
      for key in keys(ttns_probs)
        ttns_probs[key] /= ttns_sum
      end
    end
  end
  
  # Compute KL divergence: D_KL(P||Q) = sum_x P(x) * log(P(x) / Q(x))
  # where P is ground truth and Q is TTNS prediction
  let kl_divergence = 0.0
    for key in keys(probability_dict)
      p_true = probability_dict[key]
      q_ttns = get(ttns_probs, key, 0.0)
      
      if p_true > 0
        if q_ttns > 0
          kl_divergence += p_true * log(p_true / q_ttns)
        else
          # If TTNS assigns zero probability but ground truth is positive, KL divergence is infinite
          kl_divergence = Inf
          break
        end
      end
    end
    
    println("\n  Results for order=$order:")
    println("    KL divergence: $(isfinite(kl_divergence) ? round(kl_divergence, sigdigits=6) : "Inf")")
    println("    Ground truth patterns: $(length(probability_dict))")
    println("    TTNS patterns with non-zero probability: $(count(v -> v > 0, values(ttns_probs)))")
    
    return kl_divergence
  end
end

# Run for both orders
kl_div_1 = train_and_evaluate(1)
kl_div_2 = train_and_evaluate(2)

# Summary comparison
println("\n" * "="^80)
println("Summary Comparison:")
println("="^80)
println("  Order 1 KL divergence: $(isfinite(kl_div_1) ? round(kl_div_1, sigdigits=6) : "Inf")")
println("  Order 2 KL divergence: $(isfinite(kl_div_2) ? round(kl_div_2, sigdigits=6) : "Inf")")
if isfinite(kl_div_1) && isfinite(kl_div_2)
  improvement = ((kl_div_1 - kl_div_2) / kl_div_1) * 100
  println("  Improvement: $(round(improvement, sigdigits=4))%")
end
