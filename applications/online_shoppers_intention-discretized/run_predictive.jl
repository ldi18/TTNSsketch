using CSV
using DataFrames
using Random
using Graphs: nv, ne, edges, src, dst, SimpleDiGraph, add_edge!, induced_subgraph
using Statistics: mean, median
using GraphRecipes
using NetworkLayout
using Plots
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

# Train/test split (80/20) - do this FIRST to avoid data leakage
Random.seed!(1234)
all_indices = shuffle!(collect(1:nrow(df)))
train_count = Int(floor(0.8 * nrow(df)))
train_indices = all_indices[1:train_count]
test_indices_all = all_indices[(train_count+1):end]

# Balance test set by Revenue
revenue_index = findfirst(==("Revenue"), col_labels)
if revenue_index === nothing
  error("Revenue column not found")
end
revenue_zero = REPLACEMENT_RULES["Revenue"]["FALSE"]
revenue_one = REPLACEMENT_RULES["Revenue"]["TRUE"]

test_revenue_values = [col_values[revenue_index][idx] for idx in test_indices_all]
test_indices_zero = [test_indices_all[i] for i in 1:length(test_indices_all) if test_revenue_values[i] == revenue_zero]
test_indices_one = [test_indices_all[i] for i in 1:length(test_indices_all) if test_revenue_values[i] == revenue_one]

# Balance by randomly dropping from the larger class
min_count = min(length(test_indices_zero), length(test_indices_one))
if length(test_indices_zero) > min_count
  Random.seed!(1234)
  shuffle!(test_indices_zero)
  test_indices_zero = test_indices_zero[1:min_count]
elseif length(test_indices_one) > min_count
  Random.seed!(1234)
  shuffle!(test_indices_one)
  test_indices_one = test_indices_one[1:min_count]
end

test_indices = vcat(test_indices_zero, test_indices_one)
Random.seed!(1234)
shuffle!(test_indices)

println("\nTrain/test split:")
println("  Training samples: $(length(train_indices))")
println("  Test samples: $(length(test_indices)) (balanced: $(length(test_indices_zero)) Revenue=FALSE, $(length(test_indices_one)) Revenue=TRUE)")

# Build probability dicts for train and test sets
col_values_train = [col_values[j][train_indices] for j in 1:n_cols]
col_values_test = [col_values[j][test_indices] for j in 1:n_cols]

# Step 1: Compute probability dict from TRAINING dataset only (to avoid data leakage)
println("Step 1: Computing probability dict from training dataset...")
function build_joint_dict(values_by_col)
  counts = Dict{NTuple{length(values_by_col), Float64}, Int}()
  for row in 1:length(values_by_col[1])
    key = ntuple(j -> Float64(values_by_col[j][row]), length(values_by_col))
    counts[key] = get(counts, key, 0) + 1
  end
  total = sum(values(counts))
  return Dict(k => v / total for (k, v) in counts)
end

train_probability_dict = build_joint_dict(col_values_train)
test_probability_dict = build_joint_dict(col_values_test)

println("  Training unique patterns: $(length(train_probability_dict))")
println("  Test unique patterns: $(length(test_probability_dict))")
println("  Total training data points: $(length(train_indices))")
println("  Total test data points: $(length(test_indices))")

# Step 2: Use TRAINING probability dict to determine Chow-Liu tree (avoid data leakage)
println("\nStep 2: Determining Chow-Liu tree from training probability dict...")
tree = maximum_spanning_tree_recovery(train_probability_dict; max_degree=3, root_vertex=revenue_index, bmi_threshold=5e-3)
println("  Tree determined with $(nv(tree)) vertices and $(ne(tree)) edges")

# Visualize the Chow-Liu tree and extract connected component
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
  
  output_pdf_full = joinpath(@__DIR__, "chow_liu_tree_predictive.pdf")
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
  
  output_pdf_connected = joinpath(@__DIR__, "chow_liu_tree_predictive_connected.pdf")
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

# Step 3: Build sample matrix from TRAINING data only (connected vertices only)
println("\nStep 3: Building sample matrix from training data (connected vertices only)...")
# Sort connected vertices to ensure consistent column ordering
sorted_connected_vertices = sort(connected_vertices)
sample_matrix = hcat([col_values_train[j] .+ discrete_offsets[j] for j in sorted_connected_vertices]...)
println("  Sample matrix shape: $(size(sample_matrix))")

# Create vertex_input_dim only for connected vertices (using new indices)
# Use training data to determine vertex input dimensions (to avoid data leakage)
vertex_input_dim = Dict(original_to_new[i] => maximum(col_values_train[i]) + discrete_offsets[i] for i in sorted_connected_vertices)

# Helper function to build input tuple from key
function build_input_from_key(key)
  # key is a tuple of values for all original columns
  # We need to extract only the values for connected vertices and map to new indices
  input_vals = Vector{Int64}(undef, length(connected_vertices))
  for (new_idx, orig_idx) in new_to_original
    input_vals[new_idx] = Int64(key[orig_idx] + discrete_offsets[orig_idx])
  end
  return Tuple(input_vals)
end

# Function to train TTNS and evaluate for a given order
function train_and_evaluate_predictive(order::Int)
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
  
  # Step 4: Compare TTNS predictions with ground truth for train and test sets
  println("\nStep 4: Comparing TTNS predictions with ground truth (order=$order)...")
  
  function compare_predictions(prob_dict, set_name)
    comparisons = []
    for key in keys(prob_dict)
      ground_truth_prob = prob_dict[key]
      
      # Query TTNS
      input_tuple = build_input_from_key(key)
      ttns_prob = evaluate(ttns, input_tuple)
      ttns_prob = abs(ttns_prob)
      
      # Compute relative error
      if ground_truth_prob > 0
        rel_error = abs(ttns_prob - ground_truth_prob) / ground_truth_prob
      else
        rel_error = ttns_prob > 0 ? Inf : 0.0
      end
      
      push!(comparisons, (key=key, ground_truth=ground_truth_prob, ttns_pred=ttns_prob, rel_error=rel_error))
    end
    
    # Sort by ground truth probability (descending)
    sort!(comparisons, by=x -> x.ground_truth, rev=true)
    
    println("\n$set_name set - Top 20 keys (by ground truth probability):")
    println("  Key | Ground Truth | TTNS Prediction | Relative Error")
    println("  " * "-"^70)
    for (idx, comp) in enumerate(comparisons[1:min(20, length(comparisons))])
      key_str = string(comp.key)
      println("  $idx. $key_str | $(round(comp.ground_truth, sigdigits=4)) | $(round(comp.ttns_pred, sigdigits=4)) | $(round(comp.rel_error, sigdigits=4))")
    end
    
    # Summary statistics
    finite_errors = [c.rel_error for c in comparisons if isfinite(c.rel_error)]
    mean_rel_error = mean(finite_errors)
    median_rel_error = median(finite_errors)
    max_rel_error = maximum(finite_errors)
    
    println("\n$set_name set - Summary statistics:")
    println("  Total keys: $(length(comparisons))")
    println("  Mean relative error: $(round(mean_rel_error, sigdigits=4))")
    println("  Median relative error: $(round(median_rel_error, sigdigits=4))")
    println("  Max relative error: $(round(max_rel_error, sigdigits=4))")
    
    return comparisons
  end
  
  train_comparisons = compare_predictions(train_probability_dict, "Training")
  test_comparisons = compare_predictions(test_probability_dict, "Test")
  
  # Step 5: Predict Revenue for test samples
  println("\n" * "="^80)
  println("Step 5: Predicting Revenue for test samples (order=$order)")
  println("="^80)
  
  let correct_predictions = 0
    Random.seed!(1234)
    
    for (test_idx, row_idx) in enumerate(test_indices)
      # Get variable values for connected vertices only (with offsets)
      # Map original indices to new indices
      test_values_connected = Vector{Int}(undef, length(connected_vertices))
      for (new_idx, orig_idx) in new_to_original
        test_values_connected[new_idx] = col_values[orig_idx][row_idx] + discrete_offsets[orig_idx]
      end
      
      revenue_true = col_values[revenue_index][row_idx]
      revenue_true_encoded = revenue_true == revenue_zero ? revenue_zero : revenue_one
      
      # Build input tuple for Revenue=FALSE (using new indices)
      input_false = copy(test_values_connected)
      revenue_new_idx = revenue_index_connected
      input_false[revenue_new_idx] = revenue_zero + discrete_offsets[revenue_index]
      input_false_tuple = Tuple(input_false)
      
      # Build input tuple for Revenue=TRUE (using new indices)
      input_true = copy(test_values_connected)
      input_true[revenue_new_idx] = revenue_one + discrete_offsets[revenue_index]
      input_true_tuple = Tuple(input_true)
      
      # Evaluate TTNS for both Revenue values
      prob_false = abs(evaluate(ttns, input_false_tuple))
      prob_true = abs(evaluate(ttns, input_true_tuple))
      
      # Normalize to get probabilities
      total = prob_false + prob_true
      if total > 0
        p_false = prob_false / total
        p_true = prob_true / total
      else
        p_false = 0.5
        p_true = 0.5
      end
      
      # Sample from distribution
      pred_revenue = rand() < p_true ? revenue_one : revenue_zero
      
      if pred_revenue == revenue_true_encoded
        correct_predictions += 1
      end
      
      # Debug: print first few examples
      if test_idx <= 10
        println("  Test sample $test_idx: P(Revenue=FALSE)=$(round(p_false, sigdigits=4)), P(Revenue=TRUE)=$(round(p_true, sigdigits=4)), true=$(revenue_true_encoded==revenue_zero ? "FALSE" : "TRUE"), pred=$(pred_revenue==revenue_zero ? "FALSE" : "TRUE")")
      end
    end
    
    accuracy = correct_predictions / length(test_indices)
    println("\nRevenue prediction results (order=$order):")
    println("  Test samples: $(length(test_indices))")
    println("  Correct predictions: $correct_predictions")
    println("  Accuracy: $(round(accuracy, sigdigits=4))")
    
    return (train_comparisons=train_comparisons, test_comparisons=test_comparisons, accuracy=accuracy)
  end
end

# Run for both orders
results_1 = train_and_evaluate_predictive(1)
results_2 = train_and_evaluate_predictive(2)

# Summary comparison
println("\n" * "="^80)
println("Summary Comparison:")
println("="^80)
println("  Order 1 accuracy: $(round(results_1.accuracy, sigdigits=4))")
println("  Order 2 accuracy: $(round(results_2.accuracy, sigdigits=4))")
improvement = ((results_2.accuracy - results_1.accuracy) / results_1.accuracy) * 100
println("  Improvement: $(round(improvement, sigdigits=4))%")

