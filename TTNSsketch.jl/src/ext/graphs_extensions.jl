module GraphsExtensions

  using SimpleTraits
  using DataStructures: IntDisjointSets, in_same_set, union!
  using Graphs: nv, ne, src, dst, edges, edgetype, weights, IsDirected, AbstractGraph, outneighbors
  using NamedGraphs: NamedDiGraph

  export kruskal_mst, edges_in_vertex_selection

  """
  kruskal_mst(g, distmx=weights(g); minimize=true, max_degree)

  Return a vector of edges representing the minimum (by default) spanning tree of a connected, 
  undirected graph `g` with optional distance matrix `distmx` using [Kruskal's algorithm](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm).

  ### Optional Arguments
  - `minimize=true`: if set to `false`, calculate the maximum spanning tree.
  """

  function kruskal_mst end
  # see https://github.com/mauro3/SimpleTraits.jl/issues/47#issuecomment-327880153 for syntax
  @traitfn function kruskal_mst(
    g::AG::(!IsDirected), distmx::AbstractMatrix{T}=weights(g); minimize=true, max_degree::Int=typemax(Int)
  ) where {T<:Number,U,AG<:AbstractGraph{U}}
    connected_vs = IntDisjointSets(nv(g))

    mst = Vector{edgetype(g)}()
    nv(g) <= 1 && return mst
    sizehint!(mst, nv(g) - 1)

    weights = Vector{T}()
    sizehint!(weights, ne(g))
    edge_list = collect(edges(g))
    for e in edge_list
      push!(weights, distmx[src(e), dst(e)])
    end

    for e in edge_list[sortperm(weights; rev=!minimize)]
      if !in_same_set(connected_vs, src(e), dst(e))
        if max_degree >= maximum([count(e -> any((src(e), dst(e)) .== i), vcat(mst, e)) for i in 1:nv(g)])
          union!(connected_vs, src(e), dst(e))
          push!(mst, e)
        end
        (length(mst) >= nv(g) - 1) && break
      end
    end
    return mst
  end

  """
  edges_in_vertex_selection(g, selection)

  Return the number of directed edges in `g` whose endpoints both lie in `selection`.
  """
  function edges_in_vertex_selection(g::NamedDiGraph{T}, selection::AbstractVector{T})::Int where {T}
    isempty(selection) && return 0
    selected = Set(selection)
    return sum(v -> length(filter(n -> n in selected, outneighbors(g, v))), selected)
  end
end
