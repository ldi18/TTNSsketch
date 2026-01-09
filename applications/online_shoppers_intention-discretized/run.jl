using CSV
using DataFrames
using Random
using Graphs: nv, ne, edges, src, dst, SimpleDiGraph, add_edge!, induced_subgraph
using GraphRecipes
using NetworkLayout
using Plots
using LaTeXStrings
using Printf
using NamedGraphs

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
discrete_columns = String[]
continuous_columns = String[]

for (j, col_name) in enumerate(names(df))
  col_vals = df[!, col_name]
  encoded_vals = encode.(col_vals, Ref(col_labels[j]))
  
  if haskey(REPLACEMENT_RULES, col_labels[j])
    col_values[j] = Int.(encoded_vals)
    discrete_offsets[j] = minimum(col_values[j]) == 0 ? 1 : 0
    push!(discrete_columns, col_labels[j])
  else
    # Numeric columns: bin into 2 bins
    numeric_vals = Float64.(coalesce.(encoded_vals, NaN))
    valid_vals = filter(isfinite, numeric_vals)
    vmin, vmax = extrema(valid_vals)
    edges = vmin == vmax ? [vmin-0.5, vmax+0.5] : collect(range(vmin, vmax; length=3))
    col_values[j] = [clamp(searchsortedlast(edges, Float64(v)), 1, 2) for v in numeric_vals]
    discrete_offsets[j] = 0
    push!(continuous_columns, col_labels[j])
  end
end

# Print column classification
println("\nColumn Classification:")
println("  Discrete columns (categorical, no binning): $(length(discrete_columns))")
if !isempty(discrete_columns)
  println("    $(join(discrete_columns, ", "))")
end
println("  Continuous columns (numeric, binned into 2 bins): $(length(continuous_columns))")
if !isempty(continuous_columns)
  println("    $(join(continuous_columns, ", "))")
end
println()

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

num_unique_samples = length(probability_dict)
println("  Number of unique (distinct) samples in dataset: $num_unique_samples")
println("  Total data points: $(nrow(df))")

# Step 2: Use probability dict to determine Chow-Liu tree
println("\nStep 2: Determining Chow-Liu tree from probability dict...")
revenue_index = findfirst(==("Revenue"), col_labels)
if revenue_index === nothing
  error("Revenue column not found")
end

tree = maximum_spanning_tree_recovery(probability_dict; max_degree=3, root_vertex=revenue_index, bmi_threshold=1e-3)
tree_vertices = sort(collect(NamedGraphs.vertices(tree)))  # Get vertices (column indices) in the tree
println("  Tree determined with $(nv(tree)) vertices and $(ne(tree)) edges")
println("  Vertices in tree (column indices): $(join(tree_vertices, ", "))")
println("  Columns in tree: $(join([col_labels[i] for i in tree_vertices], ", "))")

# Visualize the Chow-Liu tree
function visualize_chow_liu_tree(tree, col_labels, revenue_index)
  # Set root to revenue_index to ensure proper tree structure
  rooted_tree = deepcopy(tree)
  set_root!(rooted_tree, revenue_index)
  
  # Find all reachable vertices from the root (in case of disconnected components)
  reachable_vertices = Set{Int}([revenue_index])
  function collect_reachable(v)
    for e in edges(rooted_tree)
      # Check both outgoing edges (v is source) and incoming edges (v is destination)
      if src(e) == v || dst(e) == v
        other_vertex = src(e) == v ? dst(e) : src(e)
        if !(other_vertex in reachable_vertices)
          push!(reachable_vertices, other_vertex)
          collect_reachable(other_vertex)
        end
      end
    end
  end
  collect_reachable(revenue_index)
  
  # Filter to only reachable vertices, ensuring Revenue is first (for Buchheim layout)
  reachable_set = collect(reachable_vertices)
  # Put Revenue first, then sort the rest
  other_vertices = sort([v for v in reachable_set if v != revenue_index])
  reachable_list = [revenue_index; other_vertices]
  
  # Check if we have any edges in the reachable component
  reachable_edges = [e for e in edges(rooted_tree) if src(e) in reachable_vertices && dst(e) in reachable_vertices]
  
  # ===== Plot 1: Full tree with all vertices =====
  g_full = SimpleDiGraph(nv(rooted_tree))
  for e in edges(rooted_tree)
    add_edge!(g_full, dst(e), src(e))
  end
  node_labels_full = [col_labels[i] for i in 1:nv(rooted_tree)]
  node_colors_full = [i == revenue_index ? :red : :lightblue for i in 1:nv(rooted_tree)]
  
  plt_full = plot(g_full; 
             names=node_labels_full,
             curves=false, 
             nodeshape=:circle, 
             title="Chow-Liu Tree (Full)",
             fontsize=8, 
             nodesize=0.08,
             nodecolor=node_colors_full,
             titlefontsize=14,
             aspect_ratio=1,
             linewidth=2,
             method=:spring,
             arrow=false,
             size=(1400, 1000))
  
  output_pdf_full = joinpath(@__DIR__, "chow_liu_tree.pdf")
  savefig(plt_full, output_pdf_full)
  println("  Saved full tree visualization to: $output_pdf_full")
  
  # ===== Plot 2: Only connected component with Revenue =====
  if isempty(reachable_edges)
    println("  Warning: Revenue vertex is isolated (no edges). Connected component contains only Revenue.")
    # If root is isolated, just show Revenue
    g_connected = SimpleDiGraph(1)
    node_labels_connected = [col_labels[revenue_index]]
    node_colors_connected = [:red]
    
    plt_connected = plot(g_connected; 
                 names=node_labels_connected,
                 curves=false, 
                 nodeshape=:circle, 
                 title="Chow-Liu Tree (Connected Component with Revenue)",
                 fontsize=9, 
                 nodesize=0.1,
                 nodecolor=node_colors_connected,
                 titlefontsize=14,
                 aspect_ratio=1,
                 linewidth=2,
                 method=:spring,
                 arrow=false,
                 size=(1200, 900))
  else
    # Create a mapping from original vertex indices to new indices
    # Revenue is guaranteed to be at index 1
    vertex_map = Dict(v => i for (i, v) in enumerate(reachable_list))
    
    # Verify Revenue is at index 1
    @assert vertex_map[revenue_index] == 1 "Revenue must be at index 1 for Buchheim layout"
    
    # Convert NamedDiGraph to SimpleDiGraph for plotting (only reachable vertices)
    # After set_root!, edges point child -> parent (dst is child, src is parent)
    # For Buchheim layout, we need parent -> child, so we reverse: dst -> src
    # Since set_root! ensures Revenue has no incoming edges, we can simply reverse all edges
    g_connected = SimpleDiGraph(length(reachable_list))
    for e in reachable_edges
      # Reverse edge: parent (dst) -> child (src)
      parent_idx = vertex_map[dst(e)]  # parent in rooted tree
      child_idx = vertex_map[src(e)]   # child in rooted tree
      add_edge!(g_connected, parent_idx, child_idx)
    end
    
    # Use column labels as node names (only for reachable vertices)
    node_labels_connected = [col_labels[reachable_list[i]] for i in 1:length(reachable_list)]
    
    # Highlight the root (Revenue) - it's always at index 1
    node_colors_connected = [i == 1 ? :red : :lightblue for i in 1:length(reachable_list)]
    
    plt_connected = plot(g_connected; 
                 names=node_labels_connected,
                 curves=false, 
                 nodeshape=:circle, 
                 title="Chow-Liu Tree (Connected Component with Revenue)",
                 fontsize=9, 
                 nodesize=0.1,
                 nodecolor=node_colors_connected,
                 titlefontsize=14,
                 aspect_ratio=1,
                 linewidth=2,
                 method=:buchheim,
                 arrow=false,
                 size=(1200, 900))
  end
  
  output_pdf_connected = joinpath(@__DIR__, "chow_liu_tree_connected.pdf")
  savefig(plt_connected, output_pdf_connected)
  println("  Saved connected component visualization to: $output_pdf_connected")
  println("  Note: Full tree has $(nv(tree)) vertices and $(ne(tree)) edges (may be disconnected)")
  println("  Connected component contains $(length(reachable_list)) vertices including Revenue")
  return plt_full, plt_connected, reachable_list
end

tree_plot_full, tree_plot_connected, connected_vertices = visualize_chow_liu_tree(tree, col_labels, revenue_index)

# Create connected subgraph containing only vertices reachable from Revenue
println("\nCreating connected subgraph...")
connected_tree, vertex_map_connected = induced_subgraph(tree, connected_vertices)
# Rename vertices in the subgraph to be 1, 2, 3, ... while preserving the mapping
# We need to create a new NamedDiGraph with renamed vertices
connected_tree_renamed = NamedDiGraph(length(connected_vertices))
# Create mapping from original vertex indices to new indices (1, 2, 3, ...)
original_to_new = Dict(v => i for (i, v) in enumerate(sort(connected_vertices)))
new_to_original = Dict(i => v for (v, i) in original_to_new)
# Add edges in the renamed graph
for e in edges(connected_tree)
  orig_src = src(e)
  orig_dst = dst(e)
  new_src = original_to_new[orig_src]
  new_dst = original_to_new[orig_dst]
  add_edge!(connected_tree_renamed, new_src, new_dst)
end
# Update revenue_index to the new index
revenue_index_connected = original_to_new[revenue_index]
println("  Connected subgraph has $(nv(connected_tree_renamed)) vertices and $(ne(connected_tree_renamed)) edges")
println("  Revenue is at index $revenue_index_connected in the connected subgraph")

# Step 3: Build sample matrix from all data, but only for connected vertices
println("\nStep 3: Building sample matrix (connected vertices only)...")
# Sort connected vertices to ensure consistent column ordering
sorted_connected_vertices = sort(connected_vertices)
sample_matrix = hcat([col_values[j] .+ discrete_offsets[j] for j in sorted_connected_vertices]...)
println("  Sample matrix shape: $(size(sample_matrix))")

# Create vertex_input_dim only for connected vertices (using new indices)
vertex_input_dim = Dict(original_to_new[i] => maximum(col_values[i]) + discrete_offsets[i] for i in sorted_connected_vertices)

# Count unique rows (patterns) in the sample matrix - only for columns in the connected tree
tree_sample_matrix = sample_matrix  # Already filtered to connected vertices
unique_rows = Set{Tuple{Vararg{Int}}}()
for i in 1:size(tree_sample_matrix, 1)
  push!(unique_rows, Tuple(tree_sample_matrix[i, :]))
end
n_unique_rows = length(unique_rows)
n_total_rows = size(tree_sample_matrix, 1)

# Calculate total possible patterns (product of input dimensions for vertices in connected tree only)
connected_vertex_indices = sort(collect(keys(vertex_input_dim)))  # New indices (1, 2, 3, ...)
tree_col_dims = [vertex_input_dim[i] for i in connected_vertex_indices]
total_possible_patterns = prod(tree_col_dims)

# Compute ratio
unique_ratio = n_unique_rows / total_possible_patterns

println("  Unique rows (patterns) in sample matrix (connected tree columns only): $n_unique_rows / $n_total_rows")
println("  Total possible patterns (connected tree columns only): $total_possible_patterns (product of dimensions: $(join(tree_col_dims, " × ")) = $total_possible_patterns)")
println("  Ratio of unique patterns to possible patterns: $n_unique_rows / $total_possible_patterns = $(@sprintf("%.6f", unique_ratio))")

function build_input_from_key(key)
  # key is a tuple of values for all original columns
  # We need to extract only the values for connected vertices and map to new indices
  input_vals = Vector{Int64}(undef, length(connected_vertices))
  for (new_idx, orig_idx) in new_to_original
    input_vals[new_idx] = Int64(key[orig_idx] + discrete_offsets[orig_idx])
  end
  return Tuple(input_vals)
end

# Function to train TTNS and compute KL divergence for a given order
function train_and_evaluate(order::Int)
  println("\n" * "="^80)
  println("Training TTNS with order=$order...")
  println("="^80)
  
  # Train TTNS using connected subgraph
  # vertex_to_input_pos_map maps new vertex indices (1, 2, 3, ...) to column positions in sample_matrix (also 1, 2, 3, ...)
  vertex_to_input_pos_map = Dict(i => i for i in 1:length(connected_vertices))
  ttns = Structs.TTNS(connected_tree_renamed; vertex_to_input_pos_map=vertex_to_input_pos_map, vertex_input_dim=vertex_input_dim)
  # Ensure Revenue is set as the root node (using new index)
  set_root!(ttns.tree, revenue_index_connected)
  CoreDeterminingEquations.compute_Gks!(sample_matrix, ttns; 
    sketching_kwargs=Dict(:sketching_type => Sketching.Markov, :order => order))
  println("  TTNS training complete")
  
  # Marginalize probability_dict over connected vertices only
  println("\n  Marginalizing probability distribution over connected vertices...")
  probability_dict_connected = Dict{NTuple{length(connected_vertices), Float64}, Float64}()
  for (key, prob) in probability_dict
    # Extract sub-key for connected vertices (using original indices)
    sub_key = Tuple(Float64(key[i]) for i in sorted_connected_vertices)
    probability_dict_connected[sub_key] = get(probability_dict_connected, sub_key, 0.0) + prob
  end
  println("  Marginalized distribution has $(length(probability_dict_connected)) unique patterns")
  
  # Normalize TTNS distribution
  println("\n  Computing KL divergence...")
  ttns_probs = Dict{NTuple{length(connected_vertices), Float64}, Float64}()
  let ttns_sum = 0.0
    for key in keys(probability_dict_connected)
      # key is already a tuple for connected vertices only
      input_tuple = Tuple(Int64.(key))  # Convert to Int64 tuple for evaluate
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
  # where P is ground truth (marginalized) and Q is TTNS prediction
  let kl_divergence = 0.0
    for key in keys(probability_dict_connected)
      p_true = probability_dict_connected[key]
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
    println("    Ground truth patterns (marginalized): $(length(probability_dict_connected))")
    println("    TTNS patterns: $(length(ttns_probs))")
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

