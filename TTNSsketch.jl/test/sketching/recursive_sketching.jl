# Not included in runtests.jl because no direct, i.e., elementwise comparison with non-recursive case is possible.
# We test the recursive sketching on a high level instead.

using Graphs: vertices
using ITensors: array, inds, tags
using Base: redirect_stdout, devnull

using .CoreDeterminingEquations
using .ExampleTopologies
using .GraphicalModels
using .Sketching
import .Evaluate: probability_dict

struct NonRecursiveMarkovType end

Base.getproperty(::NonRecursiveMarkovType, name::Symbol) = getproperty(Sketching.Markov, name)
Base.propertynames(::NonRecursiveMarkovType) = propertynames(Sketching.Markov)

const NonRecursiveMarkov = NonRecursiveMarkovType()

function compute_Gks_unique(ttns, f, sketching_type; order)
  sketching_kwargs = Dict{Symbol, Any}(
    :sketching_type => sketching_type,
    :sketching_set_function => Sketching.SketchingSets.MarkovCircle,
    :order => order,
  )
  svd_kwargs = Dict{Symbol, Any}()
  redirect_stdout(devnull) do
    CoreDeterminingEquations.compute_Gks!(f, ttns;
      sketching_kwargs,
      svd_kwargs,
      gauge_Gks=true,
      unique_gauge=true,
    )
  end
  return Dict(k => copy(ttns.G[k]) for k in vertices(ttns.tree))
end

tensor_to_array(t) = array(t, sort(collect(inds(t)), by=x->join(tags(x))))

@testset "Recursive Sketching" begin
  base_ttns = ExampleTopologies.ExampleTreeFromPaper()
  model = GraphicalModels.Ising_dGraphicalModel(deepcopy(base_ttns))
  prob_dict = probability_dict(model.ttns)

  recursive_ttns = deepcopy(model.ttns)
  generic_ttns = deepcopy(model.ttns)

  G_recursive = compute_Gks_unique(recursive_ttns, prob_dict, Sketching.Markov; order=1)
  G_generic = compute_Gks_unique(generic_ttns, prob_dict, NonRecursiveMarkov; order=1)

  @test sort(collect(keys(G_recursive))) == sort(collect(keys(G_generic)))
  for k in keys(G_recursive)
    @test isapprox(tensor_to_array(G_recursive[k]), tensor_to_array(G_generic[k]))
  end
end
