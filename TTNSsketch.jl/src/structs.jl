module Structs
  using NamedGraphs: NamedDiGraph, NamedEdge, vertices
  using ITensors: ITensor, Index, combiner, random_itensor, dim
  using Random: AbstractRNG, GLOBAL_RNG
  export TTNS, TTNSType, dTTNSType, cTTNSType

  abstract type TTNSType{T} end
  abstract type dTTNSType{T} <: TTNSType{T} end
  abstract type cTTNSType{T} <: TTNSType{T} end

  abstract type SketchingCoreType{T} end
  abstract type PerturbativeSketchingCoreType{T} <: SketchingCoreType{T} end
  
  """
  A structure for representing TTNS with discrete variable embeddings.
  """
  struct TTNS{T} <: dTTNSType{T}
    tree::NamedDiGraph{T}
    x_indices::Dict{T, Index}                 # Input indices x[k]
    vertex_to_input_pos_map::Dict{T, Int}     # Example: vertex (2, 1) --> input bit 3
    Q::Dict{T, ITensor}                       # Zk = U * (S V) = U * Qk, from SVD, for non-recursive evaluation
    B::Dict{T, ITensor}                       # For recursive evaluation
    G::Dict{T, ITensor}                       # Final Tensor cores of ttns
    s::Dict{T, SketchingCoreType}             # Sketching cores sk
    alpha_children_combiner::Dict{T, ITensor} # Combiner tensors to obtain joined index C(k).
                                              # Can be set once the contraction indices are defined.
    beta_inds::Dict{T, Dict{T, Index}}
    beta_inds_combiner::Dict{T, ITensor}
    beta_inds_combined::Dict{T, Index}
    gamma_inds::Dict{T, Dict{T, Index}}
    gamma_inds_combiner::Dict{T, ITensor}
    vertex_input_dim::Dict{T, Int}

    function TTNS(
        tree::NamedDiGraph{T};
        vertex_to_input_pos_map::Dict{T, Int} = T === Int ? Dict(k => k for k in keys(vertex_input_dim)) : Dict{T, Int}(),
        vertex_input_dim::Dict{T, Int} = Dict(v => 2 for v in vertices(tree))
      ) where {T}
      x_inds = Dict(k => Index(dim, "x,$(k)") for (k, dim) in vertex_input_dim)
      return new{T}(
        tree,
        x_inds,
        vertex_to_input_pos_map,
        Dict{T, ITensor}(),           # empty Qk
        Dict{T, ITensor}(),           # empty Bks
        Dict{T, ITensor}(),           # empty Gks
        Dict{T, SketchingCoreType}(), # empty sk
        Dict{T, ITensor}(),           # empty alpha_children_combiner
        Dict{T, Dict{T, Index}}(),    # empty beta_inds
        Dict{T, ITensor}(),           # empty beta_inds_combiner
        Dict{T, Index}(),             # empty beta_inds_combined
        Dict{T, Dict{T, Index}}(),    # empty gamma_inds
        Dict{T, ITensor}(),           # empty gamma_inds_combiner
        vertex_input_dim
      )
    end
  end

  """
  A structure for representing TTNS with continuous variable embeddings.
  Mirrors the discrete `TTNS` layout so that common algorithms (sketching,
  core-determining equations, perturbative machinery) can operate on both.
  """
  struct cTTNS{T} <: cTTNSType{T}
    tree::NamedDiGraph{T}
    x_indices::Dict{T, Index}                 # Input indices x[k]
    vertex_to_input_pos_map::Dict{T, Int}     # Example: vertex (2, 1) --> input bit 3
    Q::Dict{T, ITensor}                       # Zk = U * (S V) = U * Qk, from SVD
    B::Dict{T, ITensor}                       # For recursive evaluation
    G::Dict{T, ITensor}                       # Final Tensor cores of cttns
    s::Dict{T, SketchingCoreType}             # Sketching cores sk
    alpha_children_combiner::Dict{T, ITensor} # Combiner tensors to obtain joined index C(k).
                                              # Can be set once the contraction indices are defined.
    beta_inds::Dict{T, Dict{T, Index}}
    beta_inds_combiner::Dict{T, ITensor}
    beta_inds_combined::Dict{T, Index}
    gamma_inds::Dict{T, Dict{T, Index}}
    gamma_inds_combiner::Dict{T, ITensor}
    vertex_input_dim::Dict{T, Int}
    local_basis_kwargs::Dict{T, Dict{Symbol, Any}} # Dict{vertex, Dict(:T => 1, :basis_expansion_order => 3)}

    function cTTNS(
        tree::NamedDiGraph{T};
        vertex_to_input_pos_map::Dict{T, Int},
        local_basis_kwargs::Union{Dict{T, Dict{Symbol, Any}}, Dict{Symbol, Any}} = Dict{T, Dict{Symbol, Any}}()
      ) where {T}
      # Require an explicit basis specification per vertex.
      if length(local_basis_kwargs) == 0
        throw(ArgumentError("local_basis_kwargs must be non-empty."))
      end
      if isa(local_basis_kwargs, Dict{Symbol, Any})
        single = local_basis_kwargs::Dict{Symbol, Any}
        local_basis_kwargs = Dict{T, Dict{Symbol, Any}}(k => single for k in keys(vertex_to_input_pos_map))
      elseif !isa(local_basis_kwargs, Dict{T, Dict{Symbol, Any}})
        throw(ArgumentError("local_basis_kwargs must be either Dict{Symbol,Any} or Dict{T, Dict{Symbol, Any}} keyed by vertices."))
      end
      local_basis_kwargs = local_basis_kwargs::Dict{T, Dict{Symbol, Any}}

      # Ensure every vertex has an associated basis configuration.
      for k in keys(vertex_to_input_pos_map)
        if !haskey(local_basis_kwargs, k)
          throw(ArgumentError("Missing local_basis_kwargs entry for vertex $(k)."))
        end
      end

      vertex_input_dim = Dict{T, Int}()
      for k in keys(vertex_to_input_pos_map)
        if get(local_basis_kwargs[k], :discrete, false) == true
          vertex_input_dim[k] = local_basis_kwargs[k][:vertex_input_dim]
        else
          local_basis_func_set = local_basis_kwargs[k][:local_basis_func_set]
          local_basis_func_kwargs = local_basis_kwargs[k][:local_basis_func_kwargs]
          vertex_input_dim[k] = length(local_basis_func_set(local_basis_func_kwargs[:a]; local_basis_func_kwargs...))
        end
      end
      x_inds = Dict(k => Index(dim, "x,$(k)") for (k, dim) in vertex_input_dim)
      return new{T}(
        tree,
        x_inds,
        vertex_to_input_pos_map,
        Dict{T, ITensor}(),                  # empty Qk
        Dict{T, ITensor}(),                  # empty Bks
        Dict{T, ITensor}(),                  # empty Gks
        Dict{T, SketchingCoreType}(),        # empty sk
        Dict{T, ITensor}(),                  # empty alpha_children_combiner
        Dict{T, Dict{T, Index}}(),           # empty beta_inds
        Dict{T, ITensor}(),                  # empty beta_inds_combiner
        Dict{T, Index}(),                    # empty beta_inds_combined
        Dict{T, Dict{T, Index}}(),           # empty gamma_inds
        Dict{T, ITensor}(),                  # empty gamma_inds_combiner
        vertex_input_dim,
        local_basis_kwargs
      )
    end
  end

  """
  A structure for representing perturbative sketching cores sk = Ones + ε Δ.
  """
  struct PerturbativeSketchingCore{T} <: PerturbativeSketchingCoreType{T}
    v::T
    beta_inds::Dict{T, Index}
    Delta::ITensor
    function PerturbativeSketchingCore(
      v::T,
      ttns::TTNS{T},
      beta_inds::Dict{T, Index};
      rng::AbstractRNG = GLOBAL_RNG,
      ) where {T}
        inds = vcat(ttns.x_indices[v], values(beta_inds)...)
        dims = Tuple(dim.(inds))
        vals = randn(rng, Float64, dims)
        Delta = ITensor(vals, inds...)
        return new{T}(v, beta_inds, Delta)
    end
  end

end
