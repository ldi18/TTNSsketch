module Evaluate
  using ITensors #: ITensor, array, permute, 
  using Graphs: edges, vertices
  using Random
  using StatsBase: sample, ProbabilityWeights
  using Printf: @sprintf
  using ..Structs
  using ..TopologyNotation: P, depth_sorted_vertex_iterator

  export evaluate, contract_ttns, norm, sum_ttns
  export samples, probability_dict

  """
  Function to evaluate the TTNS for an input x1, x2, ..., xd.

  Works only for binary input variables.
  """
  function evaluate(ttns::dTTNSType, input_values::NTuple{N, Int64}) where {N}
    if length(input_values) != length(ttns.G)
      throw(ArgumentError("Input vector must have the same length as the number of vertices in the TTNS."))
    end
    result = 1
    for (xk_key, xk_ind) in ttns.x_indices
      x_input = ITensor(xk_ind)
      input_bit = input_values[ttns.vertex_to_input_pos_map[xk_key]]
      x_input[input_bit] = 1
      Gk_with_fixed_input = ttns.G[xk_key] * x_input
      result *= Gk_with_fixed_input
    end
    return Array(result)[1]
  end

  """
  Function to evaluate the cTTNS for an input x1, x2, ..., xd.

  contract_polynomials=false works only for binary input variables.
  """
  function evaluate(cttns::cTTNSType, input_values::NTuple{N, <:Number}; contract_polynomials=true) where {N}
    if length(input_values) != length(cttns.G)
      throw(ArgumentError("Input vector must have the same length as the number of vertices in the TTNS."))
    end
    result = 1
    for k in sort(collect(keys(cttns.x_indices)))
      input_pos = cttns.vertex_to_input_pos_map[k]
      input_value = input_values[input_pos]
      xk_ind = cttns.x_indices[k]
      is_discrete = get(cttns.local_basis_kwargs[k], :discrete, false)

      if contract_polynomials && !is_discrete
        # Continuous vertex: evaluate local basis and contract with Gk.
        local_basis_func_set = cttns.local_basis_kwargs[k][:local_basis_func_set]
        local_basis_func_kwargs = cttns.local_basis_kwargs[k][:local_basis_func_kwargs]
        local_basis_eval = local_basis_func_set(input_value; local_basis_func_kwargs...)  # length = local basis dim
        local_basis_eval_ITensor = ITensor(local_basis_eval, xk_ind)
        Gk_with_fixed_input = cttns.G[k] * local_basis_eval_ITensor
      else
        # Discrete (or polynomially disabled) vertex: insert discrete input directly.
        x_input = ITensor(xk_ind)
        input_value = Int(round(input_value))
        x_input[input_value] = 1
        Gk_with_fixed_input = cttns.G[k] * x_input
      end
      result *= Gk_with_fixed_input
    end
    return Array(result)[1]
  end

  """
  Contract all (c)ttns cores Gk in vertex_selection. If a vertex selection is provieded, only the cores in the selection and trace out all
  virtual legs. Otherwise, contract all cores. Inputs allows to fix certain input legs to a specific value (Works for binary only and operates on virtual input legs).
  """
  function contract_ttns(cttns::TTNSType; vertex_selection::Vector{T}=[], trace_open_alpha::Bool = false, inputs::Dict{Int64, Int64}=Dict{Int, Int}())::ITensor where {T}   # TODO: Specify type with Union, rename cttns to ttns?
    ITensors.disable_warn_order()
    result = 1
    for (xk_key, _) in cttns.x_indices
      if isempty(vertex_selection) || xk_key in vertex_selection   # Interpret isempty(vertex_selection) as contract all
        Gk = cttns.G[xk_key]
        if haskey(inputs, xk_key)
          x_input = ITensor(cttns.x_indices[xk_key])
          x_input[cttns.x_indices[xk_key] => 1] = inputs[xk_key]   # Direct insertion of virtual input value, no basis functions.
          Gk = Gk * x_input
        end
        result = result * Gk
      end
    end
    if trace_open_alpha
      open_legs = filter(ind -> hastags(ind, "alpha"), inds(result))
      for open_leg in open_legs
        result = result * delta(open_leg)
      end
    end
    result = permute(result, sort(collect(inds(result)), by=x -> parse(Int, string(first(tags(x))))))
    return result
  end

  """
  Sum over all possible inputs of the ttns, resulting in scalar.
  """
  function sum_ttns(ttns::dTTNSType)
    result = 1
    for (xk_key, xk_ind) in ttns.x_indices
      Gk_with_input_traced_out = ttns.G[xk_key] * delta(xk_ind)
      result *= Gk_with_input_traced_out
    end
    return Array(result)[1]
  end

  """
  Generate a matrix of samples of a discrete ttns where each sample occurs with the probability defined by the ttns.
  """
  function samples(ttns::dTTNSType, n_samples; seed::Int = 1234)::Matrix{Int}
    p_dict = probability_dict(ttns)
    Random.seed!(seed)
    rng = Random.default_rng()
    sampled_keys = sample(rng, collect(keys(p_dict)), ProbabilityWeights(collect(values(p_dict))), n_samples)
    return reduce(vcat, [collect(key)' for key in sampled_keys])
  end

  """
  Generate a dictionary of the probabilities of all possible input configurations of the ttns.
  """
  function probability_dict(ttns::TTNSType)#::Dict{NTuple{d, Int}, Float64}
    contracted_ttns = contract_ttns(ttns)
    p_dict = Dict(Tuple(ind) => contracted_ttns[ind] for ind in eachindex(contracted_ttns))
    return p_dict
  end

  """
  Compute the norm of the (c)TTNS, <TTNS|TTNS>, excluding the polynomials in the continuous case.
  """
  function norm(ttns::TTNSType; exploit_gauge::Bool = true, recompute_gauge_first::Bool = false)::Float64
    if recompute_gauge_first && exploit_gauge
      Gauging.gauge(ttns)
    end
    if exploit_gauge
      # root is isometry center
      root = findfirst(v -> isempty(P(ttns.tree, v)), vertices(ttns.tree))
      return Array(ttns.G[root] * dag(ttns.G[root]))[1]
    end
    # If reached here, contract all cores without exploiting gauge
    res = 1
    Gk_dag = Dict{Any, ITensor}(kv[1] => copy(dag(kv[2])) for kv in ttns.G)
    for level in depth_sorted_vertex_iterator(ttns.tree; include_root=true)
      for k in level
        if !isempty(P(ttns.tree, k))
          # If not root, replace the index to the parent to keep it open for the remaining cores
          ind_to_parent = first(filter(i -> hastags(i, "alpha,$(P(ttns.tree, k)[1])"), inds(ttns.G[k])))
          ind_to_parent_prime = Index(dim(ind_to_parent), tags=tags(ind_to_parent))
          replaceind!(Gk_dag[k], ind_to_parent, ind_to_parent_prime)
          replaceind!(Gk_dag[P(ttns.tree, k)[1]], ind_to_parent, ind_to_parent_prime)
        end
        res = res * (ttns.G[k] * Gk_dag[k])
      end
    end
    return Array(res)[1]
  end
end
