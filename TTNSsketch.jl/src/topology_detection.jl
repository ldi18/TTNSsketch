module TopologyDetection
  using NamedGraphs: NamedDiGraph, NamedEdge
  using Graphs, SimpleWeightedGraphs
  using ..TopologyNotation: set_root!
  using ..GraphsExtensions

  export BMI, trace_out_except, maximum_spanning_tree_recovery

  """
  Use the Chow-Liu algorithm to obtain the maximum spanning tree.

  If `bmi_threshold` is provided, edges with BMI below the threshold are excluded,
  which can create a forest. In that case, the component containing the first
  selected edge (highest BMI under the constraints) is returned.
  """
  function maximum_spanning_tree_recovery(
    P::Union{AbstractArray, Dict{NTuple{d, T}, Float64}};
    max_degree::Int=typemax(Int),
    bmi_threshold::Real=-Inf,
    root_vertex::Union{Nothing, Int}=nothing
  ) where {d, T}
    dim = isa(P, AbstractArray) ? size(P, 2) : d
    # Store all BMI values in a weighted graph. Compute BMI values efficiently with map outside of loop.
    graph = SimpleWeightedGraph(dim)
    pairs = [(i, j) for i in 1:(dim - 1) for j in (i + 1):dim]
    bmi_values = BMI.(trace_out_except.(Ref(P), first.(pairs), last.(pairs)))
    for (idx, (i, j)) in enumerate(pairs)
      weight = bmi_values[idx]
      if weight >= bmi_threshold  # This step is not included in the original Chow-Liu algorithm
        weight = weight == 0.0 ? 1e-10 : weight
        add_edge!(graph, i, j, weight)
      end
    end
    # Construct the maximum spanning tree using Kruskal's algorithm. Remove edge weights.
    maximum_spanning_tree = GraphsExtensions.kruskal_mst(graph; minimize=false, max_degree=max_degree)
    isempty(maximum_spanning_tree) && error("No edges satisfy bmi_threshold=$bmi_threshold")

    # If the threshold creates disjoint trees, select only tree including the first edge we added in algorithm above
    root_component = Set{Int}()
    if length(maximum_spanning_tree) > 1
      forest = Graphs.SimpleGraphs.SimpleGraph(dim)
      for e in maximum_spanning_tree
        add_edge!(forest, src(e), dst(e))
      end
      components = connected_components(forest)
      first_edge = first(maximum_spanning_tree)
      first_vertex = src(first_edge)
      for comp in components
        if first_vertex in comp
          root_component = Set(comp)
          break
        end
      end
      isempty(root_component) && (root_component = Set([src(first_edge), dst(first_edge)]))
    else
      first_edge = first(maximum_spanning_tree)
      root_component = Set([src(first_edge), dst(first_edge)])
    end

    maximum_spanning_tree = filter(e -> (src(e) in root_component && dst(e) in root_component), maximum_spanning_tree)
    maximum_spanning_tree = NamedDiGraph(
      Graphs.SimpleGraphs.SimpleDiGraph(Edge.([(src(e), dst(e)) for e in maximum_spanning_tree]))
    )

    # Define the hierarchy as follows:
    # 1) Find the edge with the highest weight (Always first in Kruskal result).
    # 2) Set the vertex of the two with the higher order as root.
    if root_vertex === nothing
      root_edge = first(edges(maximum_spanning_tree))
      root_vertex = src(root_edge) > dst(root_edge) ? src(root_edge) : dst(root_edge)
    end
    set_root!(maximum_spanning_tree, root_vertex)
    return maximum_spanning_tree
  end

  """
  Pxy[(x,y)] can be probabilities or counts/weights (it normalizes internally).
  """
  function BMI(Pxy::Dict{Tuple{Int,Int},Float64})
    isempty(Pxy) && return 0.0
    ks = collect(keys(Pxy))
    xs = sort(unique(first.(ks)))   # use unique values
    ys = sort(unique(last.(ks)))    # use unique values
    x_index = Dict(x => i for (i, x) in enumerate(xs))
    y_index = Dict(y => i for (i, y) in enumerate(ys))

    Pxy_mat = zeros(Float64, length(xs), length(ys))  # Use matrix for fast vectorized computation
    for ((x, y), v) in Pxy
      v < 0 && throw(ArgumentError("Negative value at ($x,$y): $v"))
      Pxy_mat[x_index[x], y_index[y]] += v
    end

    total = sum(Pxy_mat)
    total <= 0 && return 0.0
    Pxy_mat ./= total

    Px = sum(Pxy_mat, dims=2)
    Py = sum(Pxy_mat, dims=1)
    denom = Px * Py
    mask = (Pxy_mat .> 0.0) .& (denom .> 0.0)
    return sum(Pxy_mat[mask] .* log.(Pxy_mat[mask] ./ denom[mask]))
  end

  # Discretization helpers for discretization of continuous variable BMI
  _bin_index(v, edges) = clamp(searchsortedlast(edges, v), 1, length(edges)-1)

  function _find_bin_edges(vals::Vector{Float64}; bins::Int=10)
      isempty(vals) && return [0.0, 1.0]
      vmin, vmax = extrema(vals)
      vmin == vmax && return [vmin - 0.5, vmax + 0.5]
      collect(range(vmin, vmax; length=bins+1))
  end

  # Handle Continuous variable BMI discretization with type overloading
  # (Float, Int): bin X
  function BMI(Pxy::Dict{Tuple{T,Int},Float64}; bins::Int=10) where {T<:AbstractFloat}
      xs = Float64[x for ((x,_),_) in Pxy]
      xedges = _find_bin_edges(xs; bins=bins)
      Pij = Dict{Tuple{Int,Int},Float64}()
      for ((x,y), v) in Pxy
          i = _bin_index(Float64(x), xedges)
          Pij[(i,y)] = get(Pij, (i,y), 0.0) + v
      end
      return BMI(Pij)
  end

  # (Int, Float): bin Y
  function BMI(Pxy::Dict{Tuple{Int,T},Float64}; bins::Int=10) where {T<:AbstractFloat}
      ys = Float64[y for ((_,y),_) in Pxy]
      yedges = _find_bin_edges(ys; bins=bins)
      Pij = Dict{Tuple{Int,Int},Float64}()
      for ((x,y), v) in Pxy
          j = _bin_index(Float64(y), yedges)
          Pij[(x,j)] = get(Pij, (x,j), 0.0) + v
      end
      return BMI(Pij)
  end

  # (Float, Float): bin both
  function BMI(Pxy::Dict{Tuple{T1,T2},Float64}; bins::Int=10) where {T1<:AbstractFloat, T2<:AbstractFloat}
      xs = Float64[x for ((x,_),_) in Pxy]
      ys = Float64[y for ((_,y),_) in Pxy]
      xedges = _find_bin_edges(xs; bins=bins)
      yedges = _find_bin_edges(ys; bins=bins)
      Pij = Dict{Tuple{Int,Int},Float64}()
      for ((x,y), v) in Pxy
          i = _bin_index(Float64(x), xedges)
          j = _bin_index(Float64(y), yedges)
          Pij[(i,j)] = get(Pij, (i,j), 0.0) + v
      end
      return BMI(Pij)
  end

  """
  (Matrix Version - for implied probabilities) Compute a marginal distribution for i and j, i.e. trace out all dimensions except i and j from the tensor P.
  """
  function trace_out_except(samples::AbstractArray, i::Int, j::Int)
    T = eltype(samples)
    counts = Dict{Tuple{T, T}, Int}()
    total = size(samples, 1)
    for row in axes(samples, 1)
      key = (samples[row, i], samples[row, j])
      counts[key] = get(counts, key, 0) + 1
    end
    return Dict(k => v / total for (k, v) in counts)
  end

  """
  (Dict Version - for explicitly given probabilities) Compute a marginal distribution for i and j, i.e. trace out all dimensions except i and j from the tensor P.
  """
  function trace_out_except(P::Dict{NTuple{d, T}, Float64}, i::Int, j::Int) where {d, T}
    result = Dict{Tuple{T, T}, Float64}()
    for (key, value) in P
      sub_key = (key[i], key[j])
      result[sub_key] = get(result, sub_key, 0.0) + value
    end
    return result
  end

end
