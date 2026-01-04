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
using LaTeXStrings
using Random

"""
Visualize original and recovered topologies side by side.
"""
function visualize_tree_comparison(recovered_tree_order1, recovered_tree_order2, n_vertices; layout_seed=1234)
  function to_parent_child_digraph(tree)
    g = SimpleDiGraph(n_vertices)
    for e in edges(tree)
      add_edge!(g, dst(e), src(e)) # parent -> child
    end
    return g
  end

  rooted_tree_order1 = deepcopy(recovered_tree_order1)
  rooted_tree_order2 = deepcopy(recovered_tree_order2)
  set_root!(rooted_tree_order1, 1)
  set_root!(rooted_tree_order2, 1)

  rec_graph_order1 = to_parent_child_digraph(rooted_tree_order1)
  rec_graph_order2 = to_parent_child_digraph(rooted_tree_order2)
  label_width = length(string(n_vertices))
  node_labels = [lpad(string(v), label_width, '0') for v in 1:n_vertices]
  node_fontsize = 18
  node_size = 0.08
  Random.seed!(layout_seed)
  node_weights = fill(1.0, n_vertices)
  plt_rec_order1 = plot(rec_graph_order1; names=node_labels,
                        curves=false, nodeshape=:circle, title=L"\mathrm{Recovered~Topology}~(n_\mathrm{cond}=1)",
                        fontsize=node_fontsize, nodesize=node_size,
                        titlefontsize=20,
                        aspect_ratio=1,
                        linewidth=3,
                        method=:buchheim,
                        node_weights=node_weights,
                        arrow=false)
  plt_rec_order2 = plot(rec_graph_order2; names=node_labels,
                        curves=false, nodeshape=:circle, title=L"\mathrm{Recovered~Topology}~(n_\mathrm{cond}=2)",
                        fontsize=node_fontsize, nodesize=node_size,
                        titlefontsize=20,
                        aspect_ratio=1,
                        linewidth=3,
                        method=:buchheim,
                        node_weights=node_weights,
                        arrow=false)
  plt_trees = plot(plt_rec_order1, plt_rec_order2;
                   layout=(1, 2),
                   size=(1600, 800),
                   dpi=300,
                   titlefontsize=20,
                   plot_title="",
                   left_margin=2Plots.mm,
                   right_margin=2Plots.mm,
                   bottom_margin=4Plots.mm,
                   top_margin=4Plots.mm,
                   subplot_margin=2Plots.mm)
  output_pdf = joinpath(@__DIR__, "example_tree_from_thesis_markov_order_trees.pdf")
  savefig(plt_trees, output_pdf)
  println("Saved tree comparison to: $output_pdf")

  exit()
end

"""
Run ExampleTreeFromThesis with second-order conditional probabilities and
recover using second-order Markov sketching, saving a side-by-side tree figure.
"""
function run_example_tree_from_thesis(; β::Real = 1.0, order::Int = 1, seed::Int = 1234, ttns=nothing)
  ttns = ttns === nothing ? ExampleTopologies.ExampleTreeFromThesis() : ttns
  set_warn_order(length(ttns.x_indices) + 1)

  probability_dict = GraphicalModels.higher_order_probability_dict(ttns; β=β, order=order)

  recovered_tree = TopologyDetection.maximum_spanning_tree_recovery(probability_dict)

  if order != 1  # Topology detection only works reliably for conditional dependencies of order 1
    ttns_recov = deepcopy(ttns)
  else
    ttns_recov = Structs.TTNS(
      recovered_tree;
      vertex_to_input_pos_map=ttns.vertex_to_input_pos_map,
      vertex_input_dim=ttns.vertex_input_dim
    )
  end

  sketching_kwargs = Dict{Symbol, Any}(
    :sketching_type => Sketching.Markov,
    :order => order,
    :seed => seed
  )

  #CoreDeterminingEquations.compute_Gks!(probability_dict, ttns_recov; sketching_kwargs=sketching_kwargs)

  #error_stats = report_errors(probability_dict, ttns_recov, keys(probability_dict);
  #                            print_sections=false)
  # println("Max relative error: $(error_stats.overall.max)")
  return ttns_recov, probability_dict, recovered_tree
end

if abspath(PROGRAM_FILE) == @__FILE__
  println("="^60)
  println("Running ExampleTreeFromThesis with order-2 Markov sketching...")
  println("="^60)
  ttns = ExampleTopologies.ExampleTreeFromThesis()
  recovered_trees = Dict{Int, Any}()
  for order in 1:2
    println("\n" * "="^60)
    println("Running with order=$order...")
    println("="^60)
    _, _, recovered_tree = run_example_tree_from_thesis(order=order, ttns=ttns)
    recovered_trees[order] = recovered_tree
  end
  visualize_tree_comparison(recovered_trees[1], recovered_trees[2], nv(ttns.tree))
end
