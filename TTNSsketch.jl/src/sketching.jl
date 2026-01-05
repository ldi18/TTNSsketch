module Sketching
  using ITensors: ITensor
  using ..Structs: TTNSType
  using ..TopologyNotation: P, L
  import Statistics: mean

  export compute_Zk, compute_Zwk, Markov
  export ThetaSketchingFuncs, SketchingSets

  # Note: The fill_marginal_distribution_tensor! has a type overloading in the first argument:
  #  - For Dict{<:Tuple{Vararg{Int64}}, Float64} we jump in the discrete case
  #  - For Dict{<:Tuple{Vararg{Float64}}, Float64} we jump in the continuous case

  # discrete case
  """
  Sketching function discrete embedding and Dict input.
  """
  function fill_marginal_distribution_tensor!(f::Dict{<:Tuple{Vararg{Int64}}, Float64}, M_tensor::ITensor, vertex_selection::Vector{T}, ttns::TTNSType{T}; kwargs...) where {T}
    return fill_marginal_distribution_tensor_discrete!(f, M_tensor, vertex_selection, ttns; kwargs...)
  end
  function fill_marginal_distribution_tensor_discrete!(f::Dict{<:Tuple{Vararg{<:Number}}, Float64}, M_tensor::ITensor, vertex_selection::Vector{T}, ttns::TTNSType{T}; kwargs...) where {T}
    sketched_bits = [ttns.vertex_to_input_pos_map[v] for v in vertex_selection]
    default_value = isempty(f) ? 0.0 : sum(values(f)) / length(f)

    for inds in eachindex(M_tensor)
      keys_matching_the_pattern = filter(kv -> kv[1][sketched_bits] == Tuple(inds), f)
      values_of_these_keys = values(keys_matching_the_pattern)
      if isempty(values_of_these_keys)
        M_tensor[inds] = default_value
      else
        M_tensor[inds] = mean(values_of_these_keys)
      end
    end
  end

  """
  Sketching function discrete embedding and Matrix input.

  - Sample version of the function is only tested for order=1.
  """
  function fill_marginal_distribution_tensor!(f::Matrix{Int64}, M_tensor::ITensor, vertex_selection::Vector{T}, ttns::TTNSType{T}; kwargs...) where {T}
    sketched_bits = [ttns.vertex_to_input_pos_map[v] for v in vertex_selection]
    samples_sketched_part = [f[i, sketched_bits] for i in axes(f, 1)]
    total_samples = size(f, 1)
    traced_out_vertices = setdiff(collect(keys(ttns.vertex_input_dim)), vertex_selection)
    traced_out_dim = prod(ttns.vertex_input_dim[v] for v in traced_out_vertices; init=1.0)
    for inds in eachindex(M_tensor)
      match_count = count(x -> x == collect(Tuple(inds)), samples_sketched_part)
      M_tensor[inds] = match_count / (total_samples * traced_out_dim)
    end
  end

  # continuous case
  """
  Standard behaviour: Returns the coefficient tensor theta for the product basis over all sites in 
                      vertex_selection as an ITensor (isempty(filtered) == true).

  Special case:       If we keep some indices as discrete, we create theta for every combination of discrete indices.
                      Not yet supported (isempty(filtered) == false).
  """
  function fill_marginal_distribution_tensor!(f::Dict{<:Tuple{Vararg{Float64}}, Float64}, M_tensor::ITensor, vertex_selection::Vector{T}, cttns::TTNSType{T}; kwargs...)::ITensor where {T}
    # Filter discrete vertices from vertex selection
    discrete_vertices_in_selection_pos = findall(v -> get(cttns.local_basis_kwargs[v], :discrete, false) == true, vertex_selection)
    continuous_vertices_in_selection_pos = setdiff(1:length(vertex_selection), discrete_vertices_in_selection_pos)
    discrete_vertices_in_selection = vertex_selection[discrete_vertices_in_selection_pos]
    continuous_vertices_in_selection = vertex_selection[continuous_vertices_in_selection_pos]
    discrete_inputs_pos = map(v -> cttns.vertex_to_input_pos_map[v], discrete_vertices_in_selection)
    inv_perm = invperm(vcat(discrete_vertices_in_selection_pos, continuous_vertices_in_selection_pos))

    if isempty(discrete_vertices_in_selection)
      # Standard behaviour: all continuous
      theta_sketch_func = get(kwargs, :theta_sketch_function, ThetaSketchingFuncs.theta_ls)
      t = theta_sketch_func(f, vertex_selection, cttns)
      for inds in eachindex(M_tensor)
        M_tensor[inds] = t[inds]
      end
    elseif isempty(continuous_vertices_in_selection)
      # Special case: all discrete, fall back to discrete case for this vertex selection
      fill_marginal_distribution_tensor_discrete!(f, M_tensor, vertex_selection, cttns; kwargs...)
    else
      default_value = isempty(f) ? 0.0 : sum(values(f)) / length(f)
      theta_sketch_func = get(kwargs, :theta_sketch_function, ThetaSketchingFuncs.theta_ls)

      # Iterate over all combinations of discrete indices
      discrete_dims = [cttns.vertex_input_dim[v] for v in discrete_vertices_in_selection]
      for discrete_part_ind in Iterators.product((1:d for d in discrete_dims)...)
        keys_matching = filter(kv -> kv[1][discrete_inputs_pos] == Tuple(discrete_part_ind), f)
        if isempty(keys_matching)
          M_discrete_part = default_value
          # Fall back to unconditional continuous fit
          M_continuous_part = theta_sketch_func(f, continuous_vertices_in_selection, cttns)
        else
          M_discrete_part = sum(values(keys_matching))
          # Conditional distribution of continuous variables given this discrete configuration
          f_cond = Dict(k => v / M_discrete_part for (k, v) in keys_matching)
          M_continuous_part = theta_sketch_func(f_cond, continuous_vertices_in_selection, cttns)
        end
        # Join discrete and continuous parts
        for continuous_part_ind in CartesianIndices(size(M_continuous_part))
          joined = vcat(collect(discrete_part_ind), collect(Tuple(continuous_part_ind)))[inv_perm]
          M_tensor[CartesianIndex(Tuple(joined))] = M_discrete_part * M_continuous_part[continuous_part_ind]
        end
      end
    end
    return M_tensor
  end

  module SketchingSets
    using ...TopologyNotation
    using Graphs: induced_subgraph, connected_components
    export MarkovCircle,
           connected_subgraphs_containing,
           all_subgraphs_excluding_k
    """
    Returns vertex selection MarkovCircle (Eq. 35) used for order d sketching of the environment of vertex k, 
    split into left environment L and right environment R.
    """
    function MarkovCircle(ttns, k::T; sketching_kwargs) where {T}
      L_env = L_depth_d(ttns.tree, k, sketching_kwargs[:order]) # = Children for d=1
      R_env = R_depth_d(ttns.tree, k, sketching_kwargs[:order]) # = Parent for d=1
      return (L_env, R_env)
    end

    """
    Return all vertex subsets S of the vertex set excluding k such that
    S together with k forms a connected subgraph of the underlying tree.
    """
    function connected_subgraphs_containing(ttns, k::T) where {T}
      g = ttns.tree
      others = setdiff(collect(keys(ttns.x_indices)), [k])
      # Check if S ∪ {k} is a connected subgraph
      is_connected(S::Vector{T}) = begin
        vertices = vcat(S, [k])
        sg, _ = induced_subgraph(g, vertices)
        return length(connected_components(sg)) == 1
      end
      # Generate all subsets and filter to connected ones
      all_subsets = [[others[i] for i in 1:length(others) if (mask >> (i - 1)) & 0x1 == 1]
                     for mask in 1:(2^length(others) - 1)]
      return [sort(S) for S in all_subsets if is_connected(S)]
    end

    """
    Return all vertex subsets S of the vertex set excluding k (both connected and non-connected).
    Includes the empty set.
    """
    function all_subgraphs_excluding_k(ttns, k::T) where {T}
      others = setdiff(collect(keys(ttns.x_indices)), [k])
      # Generate all subsets (no connectivity filter), including empty set
      all_subsets = [[others[i] for i in 1:length(others) if (mask >> (i - 1)) & 0x1 == 1]
                     for mask in 0:(2^length(others) - 1)]
      return [sort(S) for S in all_subsets]
    end
  end

  module Markov
    using ITensors: ITensor, Index, delta, dim, dense, replaceind!, replaceinds!, dag
    using ...Structs: TTNSType
    using ...TopologyNotation: P, C, L, L_depth_d
    using ..Sketching: fill_marginal_distribution_tensor!, SketchingSets
    export compute_Zk, compute_Zwk, compute_Awk_recursive

    """
    Compute a sketch Zk of the sample distribution around vertex k, using an order d Markov sketch (Eq. 19).
    """
    function compute_Zk(f::Union{Matrix{Int64}, Dict{<:Tuple, Float64}}, ttns::TTNSType{T}, k::T; sketching_kwargs) where {T}
      Sk = get(sketching_kwargs, :sketching_set_function, SketchingSets.MarkovCircle)
      (L_env, R_env) = Sk(ttns, k; sketching_kwargs)

      vertex_selection = vcat(L_env, [k], R_env)
      betak_dims = [ttns.vertex_input_dim[l] for l in L_env]
      gammak_dims = [ttns.vertex_input_dim[r] for r in R_env]

      # Initialize Zk
      ttns.beta_inds[k] = Dict{T, Index}(l => Index(betak_dims[i], "beta,$(k),$(l)") for (i, l) in enumerate(L_env))
      xk_ind = [ttns.x_indices[k]]
      ttns.gamma_inds[k] = Dict{T, Index}(r => Index(gammak_dims[i], "gamma,$(k),$(r)") for (i, r) in enumerate(R_env))
      Zk = ITensor([ttns.beta_inds[k][l] for l in L_env]..., xk_ind, [ttns.gamma_inds[k][r] for r in R_env]...)
      fill_marginal_distribution_tensor!(f, Zk, vertex_selection, ttns; sketching_kwargs...)   # beta_{k, L(k)}, x_k, gamma_{k, R(k)}
      return Zk
    end

    """
    Compute a sketch Z_{w->k} of the sample distribution around vertex k, using a first order d (Eq. 17).
    Use recursive sketching instead for faster computation.
    """
    function compute_Zwk(f::Union{Matrix{Int64}, Dict{<:Tuple, Float64}}, ttns::TTNSType{T}, w::T; sketching_kwargs) where {T}
      Sk = get(sketching_kwargs, :sketching_set_function, SketchingSets.MarkovCircle)
      k = P(ttns.tree, w)[1]
      L_env = Sk(ttns, k; sketching_kwargs)[1]
      w_and_L_env = intersect(vcat(L(ttns.tree, w), w), L_env) # (w ∪ L(w)) ∩ (Sk ∩ L(k))   (Eq. 34), = child w for d=1
      R_env = Sk(ttns, w; sketching_kwargs)[2]                 #  Sw ∩ R(w)                 (Eq. 34), = node k  for d=1
      
      vertex_selection = vcat(w_and_L_env, R_env)
      beta_inds = [ttns.beta_inds[k][l] for (i, l) in enumerate(w_and_L_env)]
      gamma_inds = [ttns.gamma_inds[w][r] for (i, r) in enumerate(R_env)]

      Zwk = ITensor(beta_inds..., gamma_inds...)
      fill_marginal_distribution_tensor!(f, Zwk, vertex_selection, ttns; sketching_kwargs...)   # beta_{w, k}, gamma_{w, k}
      return Zwk
    end

    """
    Applies the single site sketching sw core from Eq. 27 for Markov sketching to Bk. Allows recursive sketching
    by removing children C(w) from the vertex selection (by tracing them out in Markov case). Expects parent exists.
    In the current Markov definition only the first order satisfies the recursive property
    (Right environment linked by alpha is problem in higher order case.)
    """
    function compute_Awk_recursive(ttns::TTNSType{T}, w::T; sketching_kwargs) where {T}
      if sketching_kwargs[:order] != 1
        error("Recursive sketching only implemented for order 1.")
      end
      Awk = dag(ttns.beta_inds_combiner[w]) * ttns.B[w]
      # Now apply sketching core sw (Effectively a reduction of the vertex selection in the Markov case)
      x_w = ttns.x_indices[w]
      beta_w_to_k = ttns.beta_inds[P(ttns.tree, w)[1]][w]
      inds_to_keep = L_depth_d(ttns.tree, w, sketching_kwargs[:order]-1)
      inds_to_trace = setdiff(L_depth_d(ttns.tree, w, sketching_kwargs[:order]), inds_to_keep)
      replaceind!(Awk, x_w, beta_w_to_k) # Interpret x_w as beta_{w, k}
      if !isempty(inds_to_keep)
        # Relabel beta_{l, w} to beta_{l, k} for all l in inds_to_keep
        beta_inds_to_keep = [ttns.beta_inds[w][l] for l in inds_to_keep]
        beta_inds_to_keep_replacements = [ttns.beta_inds[P(ttns.tree, w)[1]][l] for l in inds_to_keep]
        replaceinds!(Awk, beta_inds_to_keep, beta_inds_to_keep_replacements)
      end
      if !isempty(inds_to_trace)
        # Trace out beta_{l, w} for all l in inds_to_trace
        beta_inds_to_trace = [ttns.beta_inds[w][l] for l in inds_to_trace]     # Inds to all descendants of w
        trace_out_w_descendants = dense(prod(delta.(beta_inds_to_trace)))
        trace_out_w_descendants *= inv(prod(dim.(beta_inds_to_trace)))         # mean of means
        Awk *= trace_out_w_descendants
      end
      return Awk
    end
  end

  module Perturbative
    using ITensors: ITensor, random_itensor, array, Index, inds, hastags, tags, delta, replacetags, replacetags!, dim, dag, dense, replaceind!, permute, id, permutedims
    using Graphs: Edge
    using Random: MersenneTwister
    using ..Sketching: fill_marginal_distribution_tensor!
    using ..Sketching.SketchingSets: MarkovCircle,
                                     connected_subgraphs_containing,
                                     all_subgraphs_excluding_k
    using ...Structs: TTNSType, PerturbativeSketchingCore, PerturbativeSketchingCoreType
    using ...TopologyNotation: N_depth_d, L, R, N, C, P
    using ...GraphsExtensions: edges_in_vertex_selection
    export compute_Zk, compute_Zwk    
    
    const RNG = Ref{Union{Nothing, MersenneTwister}}(nothing)

    """
    Compute a sketch Zk of the sample distribution around vertex k, using perturbative sketching (Eq. 38).
    We overlay marginals of increasing order d, weighted by ε^l.
    """
    function compute_Zk(f::Union{Matrix{Int64}, Dict{<:Tuple, Float64}}, ttns::TTNSType{T}, k::T; sketching_kwargs) where {T}
      # Initialize sketching cores if needed
      if isempty(ttns.s)
        isnothing(RNG[]) && (RNG[] = MersenneTwister(get(sketching_kwargs, :seed, 1234)))
        beta_dim = get(sketching_kwargs, :beta_dim, 2)
        beta_inds_dict = Dict{T, Dict{T, Index}}()
        for v in keys(ttns.x_indices)
          beta_inds_dict[v] = Dict{T, Index}(c => Index(beta_dim, "beta,$(v),$(c)") for c in C(ttns.tree, v))
        end
        # Symmetrize parent-child beta indices
        for v in keys(ttns.x_indices)
          isempty(P(ttns.tree, v)) || (beta_inds_dict[v][P(ttns.tree, v)[1]] = beta_inds_dict[P(ttns.tree, v)[1]][v])
          ttns.s[v] = PerturbativeSketchingCore(v, ttns, beta_inds_dict[v]; rng=RNG[])
        end
      end
      
      order = get(sketching_kwargs, :order, length(ttns.x_indices)-1)
      epsilon = get(sketching_kwargs, :epsilon, 1.0)
      if get(sketching_kwargs, :use_expansion, false)
        # Expansion: Expand the perturbation tensors sk = Ones + epsilon Delta_i. Apply Delta_i on all sites in S with matching cardinality l.
        S_all = all_subgraphs_excluding_k(ttns, k)
        Zk = _compute_ZkS(f, ttns, k, S_all[1]; sketching_kwargs)
        for S in S_all[2:end]
          if length(S) <= order
            l = length(S)
            Zk += epsilon^l * _compute_ZkS(f, ttns, k, S; sketching_kwargs)
          end
        end
      else
        # No expansion: Apply perturbation tensors sk = Ones + epsilon Delta_i directly on all sites
        order = length(ttns.x_indices)-1  # Ignore order argument for non-expansion case
        Sk = get(sketching_kwargs, :sketching_set_function, MarkovCircle)
        S_sel = vcat(Sk(ttns, k; sketching_kwargs=Dict(:order => order))...)
        Zk = _compute_ZkS(f, ttns, k, S_sel; sketching_kwargs)
      end
      # Rename beta -> gamma for parent connection
      ttns.beta_inds[k] = ttns.s[k].beta_inds
      isempty(P(ttns.tree, k)) || replacetags!(Zk, "beta" => "gamma"; tags="$(k),$(P(ttns.tree, k)[1])")
      ttns.gamma_inds[k] = Dict(p => first(filter(ind -> hastags(ind, "gamma"), inds(Zk))) for p in P(ttns.tree, k))
      return Zk
    end

    """
    Compute ZkS
    """
    function _compute_ZkS(f::Union{Matrix{Int64}, Dict{<:Tuple, Float64}}, ttns::TTNSType{T}, k::T, S::Vector{T}; sketching_kwargs) where {T}
      epsilon = get(sketching_kwargs, :epsilon, 1.0)
      vertex_selection = vcat(S, k)
      
      # Initialize marginal distribution tensor
      ZkS = ITensor([ttns.x_indices[i] for i in vertex_selection]...)
      fill_marginal_distribution_tensor!(f, ZkS, vertex_selection, ttns)
      inner_beta_inds = collect(values(ttns.s[k].beta_inds))
      if isempty(S)
        return ZkS * ITensor(1.0, inner_beta_inds...)
      end

      # Compute outer beta indices to trace out
      all_beta_inds = [ind for v in S for ind in values(ttns.s[v].beta_inds)]
      complement = setdiff(keys(ttns.x_indices), S)
      beta_del_S = filter(ind -> any(hastags(ind, "$(v)") for v in complement), all_beta_inds)
      outer_beta_inds = setdiff(beta_del_S, inner_beta_inds)
      
      # Apply perturbation tensors
      for i in S
        if get(sketching_kwargs, :use_expansion, false)
          # Expansion: Sites with Delta_i, other sites with Ones are implicitly handles by marginalization
          Delta_i = ttns.s[i].Delta
        else
          # No expansion: Apply sk = Ones + epsilon Delta_i directly on all sites
          Delta_i = epsilon * ttns.s[i].Delta + ITensor(1.0, inds(ttns.s[i].Delta))
        end
        indices_to_trace = intersect(collect(values(ttns.s[i].beta_inds)), outer_beta_inds)
        if !isempty(indices_to_trace)
          Delta_i *= (dense(prod(delta.(indices_to_trace))) / prod(dim.(indices_to_trace); init=1.0))
        end
        shared_indices = intersect(inds(Delta_i), inds(ZkS))
        ZkS *= (Delta_i / prod(dim.(shared_indices); init=1.0))
      end
      return ZkS * ITensor(1.0, setdiff(inner_beta_inds, inds(ZkS)))
    end

    """Compute sketching core s_w = 1 + ε Δ_w for perturbative sketching."""
    function _compute_sw(ttns::TTNSType{T}, w::T; sketching_kwargs) where {T}
      epsilon = get(sketching_kwargs, :epsilon, 1.0)
      return epsilon * ttns.s[w].Delta + ITensor(1.0, inds(ttns.s[w].Delta)...)
    end

    """
    Compute A_{w->k} = s_w B_w for perturbative sketching (Eq. 28).
    """
    function compute_Awk_recursive(ttns::TTNSType{T}, w::T; sketching_kwargs) where {T}
      sw = _compute_sw(ttns, w; sketching_kwargs)
      Bw = dag(ttns.beta_inds_combiner[w]) * ttns.B[w]
      shared_indices = intersect(inds(sw), inds(Bw))
      return (sw * Bw) / prod(dim.(shared_indices); init=1.0)
    end
  end

  """
  Theta coefficient tensor approximation for the continuous case
  """
  module ThetaSketchingFuncs
    using ITensors: ITensor, array
    using LinearAlgebra: I
    using ...Structs: TTNSType

    "Obtain length of interval [a, b] of local basis function at vertex"
    function _T(cttns, vertex)
      return cttns.local_basis_kwargs[vertex][:local_basis_func_kwargs][:b] - cttns.local_basis_kwargs[vertex][:local_basis_func_kwargs][:a]
    end

    """
    Returns the coefficient tensor for the product basis over all sites in vertex_selection as an Array, obtained by a least squared fit.
    """
    function theta_ls(f::Dict{<:Tuple{Vararg{Float64}}, Float64}, vertex_selection::Vector{T}, cttns::TTNSType{T})::Array where {T}
      sketched_bits = [cttns.vertex_to_input_pos_map[v] for v in vertex_selection]
      local_basis_func_sets = [cttns.local_basis_kwargs[v][:local_basis_func_set] for v in vertex_selection]
      local_basis_func_kwargs = [cttns.local_basis_kwargs[v][:local_basis_func_kwargs] for v in vertex_selection]
      local_basis_dim = Tuple([cttns.vertex_input_dim[v] for v in vertex_selection])
      continuous_vertices = filter(v -> !get(cttns.local_basis_kwargs[v], :discrete, false), keys(cttns.x_indices))
      continuous_vertices_in_selection = intersect(vertex_selection, continuous_vertices)
      complement_sketched_bits = collect(setdiff(continuous_vertices, continuous_vertices_in_selection))
      integrals_sketched_bits_complement = prod(map(v -> (_T(cttns, v))^(-1/2), complement_sketched_bits); init=1.0)
      
      xs = collect(keys(f))
      N = length(xs)               # Number of samples
      M = prod(local_basis_dim)    # Number of local basis functions
      A = zeros(Float64, N, M)     # Design matrix A (N × M)
      y = zeros(Float64, N)
      function product_basis_evaluated(x::Tuple{Vararg{Float64}})
        local_bases_evaluated = map((basis_func, kwargs, bit) -> basis_func(x[bit]; kwargs...), local_basis_func_sets, local_basis_func_kwargs, sketched_bits)
        return reduce(kron, reverse(local_bases_evaluated)) # Return product basis of subspace of sketched bits, evaluated for sketched bits in x
      end    
      A = hcat(product_basis_evaluated.(xs)...)'
      y = map(x -> f[x] / integrals_sketched_bits_complement, xs)
      # Solve least squares: minimize ||A*vec(theta) - y||²
      theta_hat = A \ y
      theta_hat = reshape(theta_hat, local_basis_dim)
      return theta_hat
    end

    """
    Returns the coefficient tensor for the product basis over all sites in vertex_selection
    using a ridge-regularized least squares fit to avoid singular systems.
    """
    function theta_ls_ridge(f::Dict{<:Tuple{Vararg{Float64}}, Float64}, vertex_selection::Vector{T}, cttns::TTNSType{T}; lambda::Real=1e-6)::Array where {T}
      sketched_bits = [cttns.vertex_to_input_pos_map[v] for v in vertex_selection]
      local_basis_func_sets = [cttns.local_basis_kwargs[v][:local_basis_func_set] for v in vertex_selection]
      local_basis_func_kwargs = [cttns.local_basis_kwargs[v][:local_basis_func_kwargs] for v in vertex_selection]
      local_basis_dim = Tuple([cttns.vertex_input_dim[v] for v in vertex_selection])
      continuous_vertices = filter(v -> !get(cttns.local_basis_kwargs[v], :discrete, false), keys(cttns.x_indices))
      continuous_vertices_in_selection = intersect(vertex_selection, continuous_vertices)
      complement_sketched_bits = collect(setdiff(continuous_vertices, continuous_vertices_in_selection))
      integrals_sketched_bits_complement = prod(map(v -> (_T(cttns, v))^(-1/2), complement_sketched_bits); init=1.0)

      xs = collect(keys(f))
      N = length(xs)
      M = prod(local_basis_dim)
      A = zeros(Float64, N, M)
      y = zeros(Float64, N)
      function product_basis_evaluated(x::Tuple{Vararg{Float64}})
        local_bases_evaluated = map((basis_func, kwargs, bit) -> basis_func(x[bit]; kwargs...), local_basis_func_sets, local_basis_func_kwargs, sketched_bits)
        return reduce(kron, reverse(local_bases_evaluated))
      end
      A = hcat(product_basis_evaluated.(xs)...)'
      y = map(x -> f[x] / integrals_sketched_bits_complement, xs)

      lambda <= 0 && error("lambda must be positive for ridge regression.")
      AtA = A' * A
      Atb = A' * y
      theta_hat = (AtA + lambda * I) \ Atb
      theta_hat = reshape(theta_hat, local_basis_dim)
      return theta_hat
    end
  end

  function compute_Zk(f::Union{Matrix{Int64}, Dict{<:Tuple, Float64}}, ttns::TTNSType{T}, k::T; sketching_kwargs) where {T}
    return sketching_kwargs[:sketching_type].compute_Zk(f, ttns, k; sketching_kwargs)
  end

  function compute_Zwk(f::Union{Matrix{Int64}, Dict{<:Tuple, Float64}}, ttns::TTNSType{T}, w::T; sketching_kwargs) where {T}
    return sketching_kwargs[:sketching_type].compute_Zwk(f, ttns, w; sketching_kwargs)
  end

end
