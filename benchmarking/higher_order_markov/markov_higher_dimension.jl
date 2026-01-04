include(joinpath(@__DIR__, "..", "..", "TTNSsketch.jl", "src", "TTNSsketch.jl"))
using .TTNSsketch
using .TTNSsketch.ErrorReporting: report_errors
using .TTNSsketch.TopologyNotation: vertices, C
using .TTNSsketch.Sketching: compute_Zwk
using Printf
using ITensors: set_warn_order, norm, inds, tags, array
using LinearAlgebra: norm

ttns = ExampleTopologies.SmallExampleTree(
  vertex_input_dim=Dict(
    1 => 2,
    2 => 3,
    3 => 2,
    4 => 3,
    5 => 2,
    6 => 3
  )
)

d = length(ttns.x_indices)
set_warn_order(d+1)

probability_dict = GraphicalModels.higher_order_probability_dict(ttns; order=1)

sketching_kwargs = Dict{Symbol, Any}(
  :sketching_type => Sketching.Markov,
  :order => 1,
)

ttns_recov = deepcopy(ttns)
probability_dict_sample = Dict{NTuple{length(ttns.vertex_input_dim), Int}, Float64}()
CoreDeterminingEquations.compute_Gks!(probability_dict, ttns_recov; sketching_kwargs)
probability_dict_sample = Dict(key => evaluate(ttns_recov, key) for key in keys(probability_dict))
report_errors(probability_dict, ttns_recov, keys(probability_dict); prob_precision=8, rel_precision=4)
