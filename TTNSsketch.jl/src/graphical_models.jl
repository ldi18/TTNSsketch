module GraphicalModels
  using Random
  using ITensorNetworks: ITensorNetworks, AbstractIndsNetwork
  using ITensors: Index, delta, hastags, ITensor, eachindex
  using ITensors: array, inds, removetags!
  using Base.Iterators: product
  using NamedGraphs: NamedDiGraph, edges, src, dst, NamedEdge, vertices
  using StatsBase
  using ..Structs
  using ..ContinuousVariableEmbedding
  using ..TopologyNotation
  using ..Gauging
  using ..Evaluate

  export GraphicalModelType
  export GraphicalModel, random_cGraphicalModel, Ising_dGraphicalModel
  export higher_order_probability_dict, continuous_higher_order_probability_dict

  abstract type GraphicalModelType{T} end

  """
  Structure for representing graphical models with continuous variable embeddings.

  The coefficient vector of an edge is interpreted as the vectorized matrix [A_ij], where i is connected to vertex k and j to P(k).
  """
  struct GraphicalModel{T} <: GraphicalModelType{T}
    ttns::TTNSType{T}
    gij::Dict{NamedEdge{T}, AbstractArray}

    function GraphicalModel(
      ttns::TTNSType{T},
      gij::Dict{NamedEdge{T}, AbstractArray}=Dict{NamedEdge{T}, AbstractArray}();
      gauge_Gks::Bool = true,
      unique_gauge::Bool = true
    ) where {T}
      # Initialize tensor cores G of the cttns. To do this we define a delta tensor at each node and absorb
      # the coefficients gij of the edges into the respective child vertex.

      # First create all virtual alpha indices
      all_alpha_indices = [] # Dict of virtual indices to avoid dublicates
      for k in keys(ttns.x_indices)
        for c in C(ttns.tree, k)
          # Create alpha indices pointing down to children
          if isempty(filter(i -> hastags(i, "alpha,up,$(k),$(c)"), all_alpha_indices))
            alpha_c_k = Index(ttns.vertex_input_dim[k], "alpha,up,$(k),$(c)")
            push!(all_alpha_indices, alpha_c_k)
          end
        end
        for p in P(ttns.tree, k)
          # Create alpha index pointing up to parent (These will be contracted with the coefficient tensors)
          if isempty(filter(i -> hastags(i, "alpha,down,$(k),$(p)"), all_alpha_indices))
            alpha_k_p = Index(ttns.vertex_input_dim[k], "alpha,down,$(k),$(p)")
            push!(all_alpha_indices, alpha_k_p)
          end
        end
      end
      for k in keys(ttns.x_indices)
        # Initialize the Gk tensor as a delta tensor, create a ITensor of edge coefficient and absorb it (down and up relative to coefficient tensor, not Gk!)
        alpha_down = [first(filter(i -> hastags(i, "alpha,down,$(k),$(p)"), all_alpha_indices)) for p in P(ttns.tree, k)]
        alpha_up = [first(filter(i -> hastags(i, "alpha,up,$(k),$(c)"), all_alpha_indices)) for c in C(ttns.tree, k)]
        ttns.G[k] = delta(ttns.x_indices[k], alpha_down, alpha_up...)
        if P(ttns.tree, k) != []
          p = P(ttns.tree, k)[1]
          alpha_p = first(filter(i -> hastags(i, "alpha,up,$(k),$(p)"),   all_alpha_indices)) # Connecting coefficient tensor with parent
          alpha_k = first(filter(i -> hastags(i, "alpha,down,$(k),$(p)"), all_alpha_indices)) # Connecting coefficient tensor with k
          edge_coefficient_ITensor = ITensor(gij[NamedEdge(k, p)], alpha_k, alpha_p)  # Index order: k, P(k) (or down, up)
          ttns.G[k] = ttns.G[k] * edge_coefficient_ITensor
        end
        removetags!(ttns.G[k], "up")
      end
      if gauge_Gks
        gauge!(ttns; unique=unique_gauge)
      end
      return new{T}(ttns, gij)
    end
  end

  # discrete case
  """
  Initialize a discrete graphical model with Ising pairwise potentials.
  Eq. (40) in GenMod via TTNS (https://doi.org/10.1007/s40687-023-00381-3). 
  """
  function Ising_dGraphicalModel(ttns::Structs.dTTNSType{T}; gauge_Gks::Bool=false, unique_gauge::Bool = false) where {T}
    gij = Dict{NamedEdge{T}, AbstractArray}()
    for e in edges(ttns.tree)
      gij[e] = exp.([1.0 -1.0; -1.0 1.0])  # f_ij = xi * xj
    end
    Z = 2 * (exp(1.0) + exp(-1.0))^length(edges(ttns.tree))
    res = GraphicalModel(ttns, gij; gauge_Gks=false)
    for k in keys(res.ttns.G)
      res.ttns.G[k] = res.ttns.G[k] / Z^(1/length(keys(res.ttns.G)))
    end
    if gauge_Gks
      Gauging.gauge!(res.ttns; unique=unique_gauge)
    end
    return res
  end

  """
  Initialize a discrete graphical model with pairwise potentials g_ij = x_i * x_j,
  supporting arbitrary input dimensions per vertex.
  """
  function HigherDimensional_dGraphicalModel(ttns::Structs.dTTNSType{T}; gauge_Gks::Bool=false, unique_gauge::Bool = false) where {T}
    gij = Dict{NamedEdge{T}, AbstractArray}()
    for e in edges(ttns.tree)
      dim_e1 = ttns.vertex_input_dim[e.src]
      dim_e2 = ttns.vertex_input_dim[e.dst]
      xixj = collect(1:dim_e1) * collect(1:dim_e2)'
      exp_xixj = exp.(xixj)
      gij[e] = exp_xixj  # g_ij = xi * xj
    end
    res = GraphicalModel(ttns, gij; gauge_Gks=false)
    Z = sum_ttns(res.ttns)
    for k in keys(res.ttns.G)
      res.ttns.G[k] = res.ttns.G[k] / Z^(1/length(keys(res.ttns.G)))
    end
    if gauge_Gks
      Gauging.gauge!(res.ttns; unique=unique_gauge)
    end
    return res
  end

  """
  Construct the probability dictionary using a simple ancestor path conditional probability model.
  
  Supports arbitrary input dimensions per vertex.
  
  Conditional dependencies:
  
      P(x) = P(x_root) ∏_{v≠root} P(x_v | x_ancestors(v, order))
  
  where each node v depends on its ancestors up to depth `order` along the tree path.
  The conditional probability is parameterized as:
  
      P(x_v | x_ancestors) ∝ exp(β * ∑_{a ∈ ancestors(v, order)} s_v * s_a)
  
  where s_v, s_a are spin values mapped from states according to dimension:
  - dim=2: {-1, +1}
  - dim=3: {-1, +1, +2}
  - dim=4: {-2, -1, +1, +2}
  - dim=5: {-2, -1, +1, +2, +3}
  - etc.
  
  All ancestors up to depth `order` contribute.
  """
  function higher_order_probability_dict(ttns::Structs.dTTNSType{T};
                                            β::Real = 1.0,
                                            order::Int = 1) where {T}
    order >= 1 || error("higher_order_probability_dict expects order ≥ 1; got $(order).")
  
    """
    - dim=2: {-1, +1}, dim=3: {-1, +1, +2}, ...
    """
    function state_to_spin(state::Int, dim::Int)
      if dim == 2
        return state == 1 ? -1 : 1
      elseif dim % 2 == 0
        # Even dimension: symmetric around 0
        half = dim ÷ 2
        if state <= half
          return -(half - state + 1)  # States 1..half -> -half..-1
        else
          return state - half  # States (half+1)..dim -> 1..half
        end
      else
        # Odd dimension: symmetric around 0 with extra positive value
        half = dim ÷ 2
        if state <= half
          return -(half - state + 1)  # States 1..half -> -half..-1
        elseif state == half + 1
          return 1  # Middle state -> +1
        else
          return state - half  # States (half+2)..dim -> 2..(half+1)
        end
      end
    end
    # Vertex ordering consistent with Evaluate.probability_dict: sort by vertex id.
    vertex_order = sort(collect(keys(ttns.x_indices)))
    n_vertices = length(vertex_order)
    root = findfirst(v -> isempty(P(ttns.tree, v)), vertices(ttns.tree))

    # Root prior: uniform distribution over all possible states
    root_dim = ttns.vertex_input_dim[root]
    p_root = fill(1.0 / root_dim, root_dim)
    
    # Precompute ancestor paths for all vertices (for order >= 1, this includes at least the parent)
    ancestor_paths = Dict{T, Vector{T}}(
      v => TopologyNotation.ancestor_path(ttns.tree, v, order) for v in vertex_order if v != root
    )
    
    # Unified implementation: works for both order == 1 and order >= 2
    vertex_pos = Dict{T, Int}(v => i for (i, v) in enumerate(vertex_order))
    p_dict = Dict{Tuple{Vararg{Int}}, Float64}()
    
    # Generate all possible state combinations using vertex-specific dimensions
    state_ranges = [1:ttns.vertex_input_dim[v] for v in vertex_order]
    for state_tuple in product(state_ranges...)
      x = Tuple(state_tuple)
      root_state = x[vertex_pos[root]]
      p = p_root[root_state]
      
      # Map states to spins
      spins = Dict{T, Int}(
        v => state_to_spin(x[vertex_pos[v]], ttns.vertex_input_dim[v]) for v in vertex_order
      )
      
      # Compute conditional probabilities for each non-root vertex
      for v in vertex_order
        v == root && continue
        ancestors = get(ancestor_paths, v, [])
        child_state = x[vertex_pos[v]]
        v_dim = ttns.vertex_input_dim[v]
        
        # Conditional probability: P(x_v | ancestors) ~ exp(β * ∑_{a ∈ ancestors} s_v * s_a)
        # If ancestors list is empty (e.g., all orders dropped), use uniform distribution
        if isempty(ancestors)
          p_cond = 1.0 / v_dim  # Uniform distribution when no ancestors
        else
          # Compute log probabilities for all possible states of vertex v
          log_probs = Float64[]
          for s_v_state in 1:v_dim
            s_v_spin = state_to_spin(s_v_state, v_dim)
            log_p = β * sum(spins[a] * s_v_spin for a in ancestors)
            push!(log_probs, log_p)
          end
          
          # Normalize to get proper conditional probability
          max_log = maximum(log_probs)
          probs = exp.(log_probs .- max_log)
          Z_cond = sum(probs)
          
          # Select the appropriate conditional probability based on child_state
          p_cond = probs[child_state] / Z_cond
        end
        p *= p_cond
      end
      p_dict[x] = p
    end
    
    # Normalize
    Z = sum(values(p_dict))
    p_dict = Dict(key => val / Z for (key, val) in p_dict)
    return p_dict
  end

  # continuous case
  """
  Generate a random cGraphicalModel with coefficients sampled uniformly.
  """
  function random_cGraphicalModel(cttns::Structs.cTTNSType{T}; seed=1234, kwargs...) where {T}
    # Generate a random coefficient tensors for every edge, use only coefficients with 0 <= absolute value < 1.
    Random.seed!(seed)
    gij = Dict{NamedEdge{T}, AbstractArray}()
    for e in edges(cttns.tree)
      edge_basis_dim = length(ContinuousVariableEmbedding.product_basis([0.0], [0.0]; X_kwargs=cttns.local_basis_kwargs[e.src], Y_kwargs=cttns.local_basis_kwargs[e.dst]))      
      cases = rand(Float64, edge_basis_dim) .< 0.5
      random_vector = cases .* (0.5 .+ 0.5 .* rand(Float64, edge_basis_dim)) .+ .!cases .* (-1 .+ 1.5 .* rand(Float64, edge_basis_dim))
      gij[e] = reshape(random_vector, (cttns.vertex_input_dim[e.src], cttns.vertex_input_dim[e.dst]))
    end
    return GraphicalModel(cttns, gij; kwargs...)
  end

  """
  Generate a continuous higher-order probability dictionary using basis expansions.
  """
  function continuous_higher_order_probability_dict(cttns::Structs.cTTNSType{T};
                                                   N::Int = 10000,
                                                   order::Int = 1,
                                                   seed=1234,
                                                   a::Float64 = 0.0,
                                                   b::Float64 = 1.0) where {T}
    order >= 1 || error("continuous_higher_order_probability_dict expects order ≥ 1; got $(order).")
    Random.seed!(seed)
    
    vertex_order = sort(collect(keys(cttns.x_indices)))
    n_vertices = length(vertex_order)
    vertex_to_input_key = cttns.vertex_to_input_pos_map
    vertex_to_idx = Dict(v => i for (i, v) in enumerate(vertex_order))
    random_points = [Tuple(a .+ (b - a) .* rand(n_vertices)) for _ in 1:N]
    all_paths = [[v; TopologyNotation.ancestor_path(cttns.tree, v, order)] for v in vertex_order]
    
    basis_funcs = [cttns.local_basis_kwargs[v][:local_basis_func_set] for v in vertex_order]
    basis_kwargs = [cttns.local_basis_kwargs[v][:local_basis_func_kwargs] for v in vertex_order]
    basis_func_of_vertex(v) = (x) -> basis_funcs[vertex_to_idx[v]](x; basis_kwargs[vertex_to_idx[v]]...)
    
    # Initialize with zeros for additive aggregation of path contributions
    function_values = zeros(Float64, N)
    
    # Add contributions from each path
    # Note: Overlapping paths will still create dependencies longer than individual paths
    root = findfirst(v -> isempty(P(cttns.tree, v)), vertices(cttns.tree))
    for path in all_paths
      # Skip root-only paths (empty ancestor list)
      if length(path) > 1 || path[1] != root
        path_input_keys = [vertex_to_input_key[v] for v in path]
        product_basis_at_point(point) = reduce(kron, reverse([basis_func_of_vertex(v)(point[vertex_to_idx[v]]) for v in path]))
        basis_values = hcat(product_basis_at_point.(random_points)...)
        Random.seed!(seed + hash(path))
        coefficients = 0.5 .+ 0.5 * rand(Float64, size(basis_values, 1))
        # Compute weighted basis expansion contribution
        path_contribution = vec(coefficients' * basis_values)
        # Ensure positive and normalize to have mean ~1
        path_contribution .-= minimum(path_contribution)
        path_contribution ./= (sum(path_contribution) / N + 1e-10)
        # Add contribution from this path
        function_values .+= path_contribution
      end
    end
    
    # Normalize final result
    Z = sum(function_values)
    function_values ./= Z
    return Dict(random_points[i] => function_values[i] for i in 1:N)
  end

  """
  Generate a continuous higher-order probability dictionary using basis expansions,
  using a single ancestor chain for linear topologies.
  
  Only works for linear topologies (at most one child per vertex).
  Creates a single ancestor chain starting from the first vertex in sorted order
  (smallest vertex ID), extending towards the root with growing condition order.
  For example, if vertex_order = [1,2,3,4,5,6], the paths are:
  order=1: [1,2], order=2: [1,2,3], order=3: [1,2,3,4], etc.
  This ensures the required sketching order matches the condition order.
  """
  function continuous_higher_order_probability_dict_non_overlapping_linear(cttns::Structs.cTTNSType{T};
                                                                          N::Int = 10000,
                                                                          order::Int = 1,
                                                                          seed=1234,
                                                                          a::Float64 = 0.0,
                                                                          b::Float64 = 1.0) where {T}
    order >= 1 || error("continuous_higher_order_probability_dict_non_overlapping_linear expects order ≥ 1; got $(order).")
    Random.seed!(seed)
    
    vertex_order = sort(collect(keys(cttns.x_indices)))
    n_vertices = length(vertex_order)
    vertex_to_idx = Dict(v => i for (i, v) in enumerate(vertex_order))
    random_points = [Tuple(a .+ (b - a) .* rand(n_vertices)) for _ in 1:N]
    
    # Create a single ancestor chain starting from the first vertex in sorted order
    v_start = vertex_order[1]
    path = [v_start; TopologyNotation.ancestor_path(cttns.tree, v_start, order)]
    
    basis_funcs = [cttns.local_basis_kwargs[v][:local_basis_func_set] for v in vertex_order]
    basis_kwargs = [cttns.local_basis_kwargs[v][:local_basis_func_kwargs] for v in vertex_order]
    basis_func_of_vertex(v) = (x) -> basis_funcs[vertex_to_idx[v]](x; basis_kwargs[vertex_to_idx[v]]...)
    
    # Compute product basis expansion for the single path
    product_basis_at_point(point) = reduce(kron, reverse([basis_func_of_vertex(v)(point[vertex_to_idx[v]]) for v in path]))
    basis_values = hcat(product_basis_at_point.(random_points)...)
    Random.seed!(seed)
    coefficients = 0.5 .+ 0.5 * rand(Float64, size(basis_values, 1))
    function_values = vec(coefficients' * basis_values)
    
    # Ensure positive and normalize
    function_values .-= minimum(function_values)
    function_values ./= (sum(function_values) / N + 1e-10)
    
    # Normalize final result
    Z = sum(function_values)
    function_values ./= Z
    return Dict(random_points[i] => function_values[i] for i in 1:N)
  end

end
