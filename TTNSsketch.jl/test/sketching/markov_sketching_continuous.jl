using Printf
using Test
using ITensors: set_warn_order
using Random
using .TTNSsketch: GraphicalModels, Evaluate, Sketching,
                 CoreDeterminingEquations, ExampleTopologies,
                 ErrorReporting, samples, ContinuousVariableEmbedding

# Here we test the full logic of the TTNSsketching algorithm using the Markov sketches,
# with the continous variable embedding and a non-trivial vertex-input map.
@testset "Markov TTNS sketching - continuous" begin
  T = 1.0
  local_basis_kwargs = Dict{Symbol, Any}(
    :local_basis_func_set => ContinuousVariableEmbedding.legendre_basis,
    :local_basis_func_kwargs => Dict{Symbol, Any}(:a => 0.0, :b => T, :basis_expansion_order => 2)
  )
  vertex_to_input_pos_map = Dict(1 => 3, 2 => 1, 3 => 6, 4 => 2, 5 => 5, 6 => 4)
  cttns = ExampleTopologies.SmallExampleTree(
    vertex_to_input_pos_map;
    continuous=true,
    local_basis_kwargs
  )
  d = length(cttns.x_indices)
  set_warn_order(d + 1)

  Random.seed!(1234)
  model = GraphicalModels.random_cGraphicalModel(cttns; seed=42)
  n_samples = 15000
  xs = [Tuple(T * rand(d)) for _ in 1:n_samples]
  f_vals = map(x -> Evaluate.evaluate(model.ttns, x), xs)
  f = Dict(x => f_val for (x, f_val) in zip(xs, f_vals))

  sketching_kwargs = Dict{Symbol, Any}(
    :theta_sketch_function => Sketching.ThetaSketchingFuncs.theta_ls,
    :sketching_type => Sketching.Markov,
    :sketching_set_function => Sketching.SketchingSets.MarkovCircle,
    :order => 1
  )

  cttns_recov = deepcopy(cttns)
  CoreDeterminingEquations.compute_Gks!(f, cttns_recov; sketching_kwargs, gauge_Gks=false)
  recov_vals = map(x -> Evaluate.evaluate(cttns_recov, x), xs)
  max_abs = maximum(abs.(f_vals))
  rel_errs = max_abs == 0 ? abs.(recov_vals .- f_vals) : abs.(recov_vals .- f_vals) / max_abs
  mean_err = sum(rel_errs) / length(rel_errs)
  @test mean_err < 0.2
end

