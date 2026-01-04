include(joinpath(@__DIR__, "..", "src", "TTNSsketch.jl"))

using LinearAlgebra
using Random
using Graphs: edges
using NamedGraphs: NamedEdge
using Statistics: mean
using ITensors: array, inds, tags
using .TTNSsketch

Random.seed!(1234)

T = 1
local_basis_kwargs = Dict{Symbol, Any}(
  :local_basis_func_set => ContinuousVariableEmbedding.legendre_basis,
  :local_basis_func_kwargs => Dict{Symbol, Any}(:a => 0.0, :b => T, :basis_expansion_order => 2)
)
cttns = ExampleTopologies.Linear(5; continuous=true, local_basis_kwargs)
model = GraphicalModels.random_cGraphicalModel(cttns; seed=42)
d = size(collect(cttns.vertex_to_input_pos_map))[1]
n_samples = 50000

Random.seed!(42)
xs = [Tuple(T * rand(d)) for _ in 1:n_samples]
f_vals = map(x -> evaluate(model.ttns, x), xs)
f = Dict(x => f_val for (x, f_val) in zip(xs, f_vals))

sketching_kwargs = Dict{Symbol, Any}(       # Select sketching function in sketching_kwargs.
  :sketching_type => Sketching.Markov,
  :order => 1
)
# Note: theta_sketch_function defaults to Sketching.ThetaSketchingFuncs.theta_ls if not specified

cttns_recov = deepcopy(cttns)
@time CoreDeterminingEquations.compute_Gks!(f, cttns_recov; sketching_kwargs, gauge_Gks=false)  # overwrites Gks
recov_vals = map(x -> evaluate(cttns_recov, x), xs)
rel_errs = abs.((recov_vals .- f_vals)) ./ maximum(f_vals)
println("Maximum relative error: ", maximum(rel_errs))
println("Mean relative error: ", mean(rel_errs))