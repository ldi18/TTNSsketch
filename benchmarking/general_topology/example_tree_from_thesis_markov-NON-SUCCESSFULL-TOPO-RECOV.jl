# Demonstrate that with using recovery order 3 > 2 we can weigh out non optimal tree recovery. 
# Conditional dependencies of order 2 in data, recovery order of 3 in Sketching.

include(joinpath(@__DIR__, "..", "..", "TTNSsketch.jl", "src", "TTNSsketch.jl"))
using .TTNSsketch
using .TTNSsketch.ExampleTopologies
using .TTNSsketch.GraphicalModels
using .TTNSsketch.CoreDeterminingEquations
using .TTNSsketch.ErrorReporting: report_errors
using .TTNSsketch.Sketching
using .TTNSsketch.Structs
using .TTNSsketch.TopologyDetection
using .TTNSsketch.TopologyNotation: set_root!
using ITensors: set_warn_order
using BenchmarkTools
using Graphs: edges, src, dst, vertices, add_edge!, nv, SimpleDiGraph
using GraphRecipes
using NetworkLayout
using Plots
using Random

"""
Visualize the recovered topology
"""
function visualize_recovered_topology(ttns, recovered_tree, order; layout_seed=1234)
  function to_parent_child_digraph(tree)
    g = SimpleDiGraph(nv(tree))
    for e in edges(tree)
      add_edge!(g, dst(e), src(e)) # parent -> child
    end
    return g
  end

  println("Recovered tree edges:")
  for e in edges(recovered_tree)
    println("  $(src(e)) -> $(dst(e))")
  end

  rooted_tree = deepcopy(recovered_tree)
  set_root!(rooted_tree, 1)
  rec_graph = to_parent_child_digraph(rooted_tree)
  label_width = length(string(nv(ttns.tree)))
  node_labels = [lpad(string(v), label_width, '0') for v in 1:nv(ttns.tree)]
  node_fontsize = 10
  node_size = 1.0
  Random.seed!(layout_seed)
  node_weights = fill(1.0, nv(ttns.tree))
  plt_rec = plot(rec_graph; names=node_labels,
                 curves=false, nodeshape=:circle, title="Recovered Tree",
                 fontsize=node_fontsize, nodesize=node_size, fontfamily="Courier",
                 method=:buchheim, node_weights=node_weights, arrow=false)
  plt_trees = plot(plt_rec;
                   layout=(1, 1),
                   size=(1600, 640),
                   dpi=300,
                   plot_title="",
                   left_margin=8Plots.mm,
                   right_margin=8Plots.mm,
                   bottom_margin=12Plots.mm,
                   top_margin=6Plots.mm)
  output_pdf = joinpath(@__DIR__, "example_tree_from_thesis_markov_order$(order)_trees.pdf")
  savefig(plt_trees, output_pdf)
  println("Saved tree comparison to: $output_pdf")
end

"""
Run ExampleTreeFromThesis with second-order conditional probabilities and
recover using second-order Markov sketching, saving a side-by-side tree figure.
"""
function run_example_tree_from_thesis(;
  cond_order::Int = 2, recovery_order::Int = 2,
  β::Real = 1.0, order::Int = 1, seed::Int = 1234)
  ttns = ExampleTopologies.ExampleTreeFromThesis()
  set_warn_order(length(ttns.x_indices) + 1)

  probability_dict = GraphicalModels.higher_order_probability_dict(ttns; β=β, order=cond_order)

  recovered_tree = TopologyDetection.maximum_spanning_tree_recovery(probability_dict)
  visualize_recovered_topology(ttns, recovered_tree, cond_order)

  ttns_recov = Structs.TTNS(
    recovered_tree;
    vertex_to_input_pos_map=ttns.vertex_to_input_pos_map,
    vertex_input_dim=ttns.vertex_input_dim
  )

  sketching_kwargs = Dict{Symbol, Any}(
    :sketching_type => Sketching.Markov,
    :order => recovery_order,
    :seed => seed
  )

  CoreDeterminingEquations.compute_Gks!(probability_dict, ttns_recov; sketching_kwargs=sketching_kwargs)

  error_stats = report_errors(probability_dict, ttns_recov, keys(probability_dict);
                              print_sections=false)
  println("Max relative error: $(error_stats.overall.max)")
  return ttns_recov, probability_dict
end

if abspath(PROGRAM_FILE) == @__FILE__
  println("="^60)
  println("Running ExampleTreeFromThesis with order-2 Markov sketching...")
  println("="^60)
  println("Running with cond_order=2, recovery_order=3...")
  println("="^60)
  run_example_tree_from_thesis(cond_order=2, recovery_order=3)    # Use recovery order 3 > 2 to weigh out non optimal tree recovery
end
