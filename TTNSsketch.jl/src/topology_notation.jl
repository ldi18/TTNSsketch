module TopologyNotation
  using Graphs: SimpleDiGraph, all_neighbors, inneighbors, outneighbors, vertices, has_vertex
  using NamedGraphs: NamedDiGraph
  using NamedGraphs.GraphsExtensions: rem_edge!, add_edge!
  using Graphs: Edge, is_directed, edges

  export C, P, L, R, N, N_depth_d, L_depth_d, R_depth_d, ancestor_path
  export x_ITensorIndices
  export set_root!
  export depth_sorted_vertex_iterator

  """
  Return all children of a vertex in a directed graph.
  """
  function C(graph::NamedDiGraph, vertex::T)::Vector{T} where {T}
    return inneighbors(graph, vertex)
  end

  """
  Return the parent of a vertex in a directed graph.
  """
  function P(graph::NamedDiGraph, vertex::T)::Vector{T} where {T}
    parent = outneighbors(graph, vertex)
    if length(parent) > 1
        return error("Vertex $vertex has more than one parent")
    end
    if isempty(parent)
        return Int64[]
    end
    return parent
  end

  """
  Return all descendants of a vertex in a directed graph.
  """
  function L(graph::NamedDiGraph, vertex::T)::Vector{T} where {T}
    descendants_list = Set{T}()
    current_level_children = C(graph, vertex)
    while !isempty(current_level_children)
      for child in current_level_children
        push!(descendants_list, child)
      end
      current_level_children = reduce(vcat, [C(graph, c) for c in current_level_children], init=Int[])
    end
    return sort(collect(descendants_list))
  end

  """
  Return all descendants i with dist(k, i) <= d of a vertex k in a directed graph. 
  """
  function L_depth_d(graph::NamedDiGraph, vertex::T, depth::Int)::Vector{T} where {T}
    descendants_list = Set{T}()
    current_level_children = C(graph, vertex)
    for _ in 1:depth
      if isempty(current_level_children)
        break
      end
      for child in current_level_children
        push!(descendants_list, child)
      end
      current_level_children = reduce(vcat, [C(graph, c) for c in current_level_children], init=Int[])
    end
    return sort(collect(descendants_list))
  end

  """
  Return all non-descendants of a vertex in a directed graph.
  """
  function R(graph::NamedDiGraph, vertex::T; ignore_siblings::Bool=false)::Vector{T} where {T}
    all_vertices = Set(vertices(graph))
    descendants = L(graph, vertex)
    non_descendants = setdiff(all_vertices, descendants)
    if ignore_siblings
      siblings_and_their_descendants = !isempty(P(graph, vertex)) ? vcat([union(s, L(graph, s)) for s in setdiff(C(graph, P(graph, vertex)[1]), vertex)]...) : []
      non_descendants = setdiff(non_descendants, siblings_and_their_descendants)
    end
    delete!(non_descendants, vertex)
    return sort(collect(non_descendants))
  end

  """
  Return all non-descendants i with dist(k, i) <= d of a vertex k in a directed graph. 
  """
  function R_depth_d(graph::NamedDiGraph, vertex::T, depth::Int; ignore_siblings::Bool=false)::Vector{T} where {T}
    non_descendants_list = Set{T}()
    if depth < 1 || isempty(P(graph, vertex))
      return []
    end
    current_level_parent = P(graph, vertex)[1]
    push!(non_descendants_list, current_level_parent)
    skip_vertex = vertex  # to avoid step back to the vertex itself
    for d in 2:depth
      if d > 2 || !ignore_siblings
        # Only track children of grandparents, but not parents
        # (We iterate over children in TTNSsketch to consider them all)
        current_level_other_children = setdiff(C(graph, current_level_parent), [skip_vertex])
        if !isempty(current_level_other_children)
          push!(non_descendants_list, current_level_other_children...)
        end
        descendants_of_current_level_other_children = reduce(vcat, [L_depth_d(graph, c, depth-2) for c in current_level_other_children], init=Int[])
        if !isempty(descendants_of_current_level_other_children)
          push!(non_descendants_list, descendants_of_current_level_other_children...)
        end
      end
      skip_vertex = current_level_parent
      if !isempty(P(graph, current_level_parent))
        current_level_parent = P(graph, current_level_parent)[1]
        push!(non_descendants_list, current_level_parent)
      else
        break
      end
    end
    return sort(collect(non_descendants_list))
  end

  """
  Return the input index x for a vertex in a TTNS struct as an ITensor Index.
  """
  function x_ITensorIndices(ttns, vertex::T)::Index where {T}
    return ttns.x_indices[vertex]
  end

  """
  Return all neighbours of a vertex in a directed graph.
  """
  function N(graph::NamedDiGraph, vertex::T)::Vector{T} where {T}
    return all_neighbors(graph, vertex)
  end

  """
  Return all depth-d neighbours of a vertex in a directed graph, i.e., all vertices
  that can be reached from the vertex by following d edges.
  """
  function N_depth_d(graph::NamedDiGraph, vertex::T, depth::Int)::Vector{T} where {T}
    depth_d_neighbors = Set{T}([vertex])
    current_level = Set([vertex])
    for _ in 1:depth
      next_level = Set{T}()
      for v in current_level
        for neighbor in setdiff(N(graph, v), depth_d_neighbors)
          push!(next_level, neighbor)
        end
      end
      union!(depth_d_neighbors, next_level)
      current_level = next_level
    end
    return sort(collect(setdiff(depth_d_neighbors, [vertex])))
  end

  """
  Return the ancestor path of a vertex up to a given depth.
  Returns a vector of ancestors starting from the parent (depth 1) up to depth d.
  """
  function ancestor_path(graph::NamedDiGraph, vertex::T, depth::Int)::Vector{T} where {T}
    ancestors = T[]
    if depth < 1 || isempty(P(graph, vertex))
      return ancestors
    end
    current = P(graph, vertex)[1]
    for _ in 1:depth
      push!(ancestors, current)
      if isempty(P(graph, current))
        break
      end
      current = P(graph, current)[1]
    end
    return ancestors
  end

  """
  Set the root in a directed graph to vertex with index `root_index`.
  """
  function set_root!(graph::NamedDiGraph, root_index::T)::Nothing where {T}
    if !is_directed(graph)
      error("Graph must be directed")
    end
    if !has_vertex(graph, root_index)
      error("Root index $root_index not in graph")
    end
    indices_previous_level = Set{T}()
    indices_current_level = Set{T}([root_index])
    while !isempty(indices_current_level)
      downstream_neighbors_current_level = Set{T}()
      for v in indices_current_level
        for n in setdiff(all_neighbors(graph, v), indices_previous_level)
          rem_edge!(graph, (v) => (n))
          add_edge!(graph, (n) => (v))
          push!(downstream_neighbors_current_level, n)
        end
      end
      (indices_current_level, indices_previous_level) = (downstream_neighbors_current_level, indices_current_level)
    end
  end

  """
  Return a vector of vectors, where the n'th inner vector contains all vertices at depth n
  (distance n from the root).
  Form of result: [[v with max depth...], [v with max depth - 1], ..., [children of root]]
  """
  function depth_sorted_vertex_iterator(graph::NamedDiGraph{T}; include_root::Bool=false)::Vector{Vector{Int64}} where {T}
    root = findfirst(v -> isempty(P(graph, v)), vertices(graph))
    vertices_in_level = [[root]]
    while true  # Define levels
      vertices_in_next_level = vcat([C(graph, p) for p in vertices_in_level[end]]...)
      if isempty(vertices_in_next_level)
        break
      end
      push!(vertices_in_level, vertices_in_next_level)
    end
    if !include_root
      vertices_in_level = vertices_in_level[2:end]
    end
    return reverse(vertices_in_level)
  end

end