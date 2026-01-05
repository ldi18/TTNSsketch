include(joinpath(@__DIR__, "..", "src", "TTNSsketch.jl"))
using .TTNSsketch
using .TTNSsketch.ErrorReporting: report_errors
using .TTNSsketch.TopologyNotation: vertices, C
using .TTNSsketch.Sketching: compute_Zwk
# Main.TTNSsketch.TopologyNotation.example_function()
using Printf
using ITensors: set_warn_order, norm, inds, tags, array
using LinearAlgebra: norm

# Define the example topology. Other topologies are available in ExampleTopologies.jl.
ttns = ExampleTopologies.ExampleTreeFromPaper()
#ttns = ExampleTopologies.ExampleTreeFromPaper( # Or also with non-trivial assignment of input variables to vertices
#  vertex_to_input_pos_map=Dict(1 => 7, 2 => 3, 3 => 9, 4 => 1, 5 => 5, 6 => 10, 7 => 2, 8 => 8, 9 => 4, 10 => 6)
#)

use_sample_matrix = false  # Either choose samples implying the probability distribution,
                           # or use the exact function values. The sample-based mode is only tested for order=1.
d = length(ttns.x_indices) # dimensions
set_warn_order(d+1)
N = 30000                  # Only used if use_sample_matrix is true.

model = GraphicalModels.Ising_dGraphicalModel(ttns)
probability_dict = Evaluate.probability_dict(model.ttns)

sketching_kwargs = Dict{Symbol, Any}(       # Select sketching function in sketching_kwargs.
  :sketching_type => Sketching.Markov,
  :order => 1
)
# sketching_kwargs = Dict{Symbol, Any}(
#   :sketching_type => Sketching.Perturbative,
#   :order => d-1,
#   :beta_dim => 2,
#   :epsilon => 1
# )

ttns_recov = deepcopy(ttns)
probability_dict_sample = Dict{NTuple{length(ttns.vertex_input_dim), Int}, Float64}()

if use_sample_matrix
  sample_matrix = TTNSsketch.samples(model.ttns, N; seed=1234)
  CoreDeterminingEquations.compute_Gks!(sample_matrix, ttns_recov; sketching_kwargs)
  sample_counts = Dict{NTuple{length(ttns.vertex_input_dim), Int}, Int}()
  for row in eachrow(sample_matrix)
    tuple_row = Tuple(row)
    sample_counts[tuple_row] = get(sample_counts, tuple_row, 0) + 1
  end
  probability_dict_sample = Dict(key => count / size(sample_matrix, 1) for (key, count) in sample_counts)
  report_errors(probability_dict, ttns_recov, keys(probability_dict); prob_precision=8, rel_precision=4)
else
  CoreDeterminingEquations.compute_Gks!(probability_dict, ttns_recov; sketching_kwargs)
  probability_dict_sample = Dict(key => evaluate(model.ttns, key) for key in keys(probability_dict))
  report_errors(probability_dict, ttns_recov, keys(probability_dict); prob_precision=8, rel_precision=4)
end
