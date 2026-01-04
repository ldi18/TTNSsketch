module ExampleTopologies

  using Graphs: binary_tree, src, dst, edges
  using Random
  using NamedGraphs: NamedDiGraph, rename_vertices, vertices
  using ITensorNetworks: ITensorNetwork, siteinds, vertices, edges, add_vertex!, add_edge!

  #using Main.TTNSsketch
  using ..Structs

  export ExampleTreeFromPaper, BinaryTree, LargerExampleTree, SmallExampleTree, ExampleTreeFromThesis

  function BinaryTree(
    depth::Int;
    vertex_to_input_pos_map::Union{Dict{Int, Int}, Nothing} = nothing,
    vertex_input_dim::Union{Dict, Nothing} = nothing,
    continuous::Bool=false,
    kwargs...
  )
    n_vertices = 2^depth - 1
    b = binary_tree(depth)
    g = NamedDiGraph(n_vertices)
    # Reverse edges: Graphs.jl uses parent->child, but we need child->parent
    # This ensures vertex 1 (root) is at the top with no outgoing edges
    for e in edges(b)
      add_edge!(g, dst(e), src(e))  # Reverse: child -> parent
    end
    if isnothing(vertex_to_input_pos_map)
      vertex_to_input_pos_map = Dict{Int, Int}(v => v for v in vertices(g))
    end
    vertex_input_dim = isnothing(vertex_input_dim) ? Dict(v => 2 for v in vertices(g)) : vertex_input_dim
    return continuous ? Structs.cTTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, kwargs...) :
           Structs.TTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, vertex_input_dim=vertex_input_dim, kwargs...)
  end

  function SmallExampleTree(
    vertex_to_input_pos_map = Dict{Int, Int}(k => k for k in 1:6);
    vertex_input_dim::Union{Dict, Nothing} = nothing,
    continuous::Bool=false,
    kwargs...
  )
    g = NamedDiGraph(6)
    add_edge!(g, 1, 2)
    add_edge!(g, 3, 2)
    add_edge!(g, 2, 4)
    add_edge!(g, 5, 4)
    add_edge!(g, 4, 6)
    vertex_input_dim = isnothing(vertex_input_dim) ? Dict(v => 2 for v in vertices(g)) : vertex_input_dim
    return continuous ? Structs.cTTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, kwargs...) :
           Structs.TTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, vertex_input_dim=vertex_input_dim, kwargs...)
  end

  function ExampleTreeFromPaper(
    vertex_to_input_pos_map = Dict{Int, Int}(k => k for k in 1:10);
    vertex_input_dim::Union{Dict, Nothing} = nothing,
    continuous::Bool=false,
    kwargs...
  )
    g = NamedDiGraph(10)
    add_edge!(g, 1, 2)
    add_edge!(g, 3, 2)
    add_edge!(g, 2, 4)
    add_edge!(g, 5, 4)
    add_edge!(g, 4, 6)
    add_edge!(g, 6, 7)
    add_edge!(g, 8, 7)
    add_edge!(g, 9, 7)
    add_edge!(g, 7, 10)
    vertex_input_dim = isnothing(vertex_input_dim) ? Dict(v => 2 for v in vertices(g)) : vertex_input_dim
    return continuous ? Structs.cTTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, kwargs...) :
           Structs.TTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, vertex_input_dim=vertex_input_dim, kwargs...)
  end

  function Linear(
    length::Int;
    vertex_input_dim::Union{Dict, Nothing} = nothing,
    continuous::Bool=false,
    kwargs...
  )
    vertex_to_input_pos_map = Dict{Int, Int}(k => k for k in 1:length)
    g = NamedDiGraph(length)
    for i in 1:(length - 1)
      add_edge!(g, i, i + 1)
    end
    vertex_input_dim = isnothing(vertex_input_dim) ? Dict(v => 2 for v in vertices(g)) : vertex_input_dim
    return continuous ? Structs.cTTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, kwargs...) :
           Structs.TTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, vertex_input_dim=vertex_input_dim, kwargs...)
  end

  function MiniMPS_R(
    vertex_to_input_pos_map = Dict{Int, Int}(k => k for k in 1:3);
    vertex_input_dim::Union{Dict, Nothing} = nothing,
    continuous::Bool=false,
    kwargs...
  )
    g = NamedDiGraph(3)
    # (1)-(2)-[3]
    add_edge!(g, 2, 3)
    add_edge!(g, 1, 2)
    vertex_input_dim = isnothing(vertex_input_dim) ? Dict(v => 2 for v in vertices(g)) : vertex_input_dim
    return continuous ? Structs.cTTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, kwargs...) :
           Structs.TTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, vertex_input_dim=vertex_input_dim, kwargs...)
  end

  function MiniMPS_L(
    vertex_to_input_pos_map = Dict{Int, Int}(k => k for k in 1:3);
    vertex_input_dim::Union{Dict, Nothing} = nothing,
    continuous::Bool=false,
    kwargs...
  )
    g = NamedDiGraph(3)
    # [1]-(2)-(3)
    add_edge!(g, 2, 1)
    add_edge!(g, 3, 2)
    vertex_input_dim = isnothing(vertex_input_dim) ? Dict(v => 2 for v in vertices(g)) : vertex_input_dim
    return continuous ? Structs.cTTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, kwargs...) :
           Structs.TTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, vertex_input_dim=vertex_input_dim, kwargs...)
  end

  function LinearWithTwoChildren(
    vertex_to_input_pos_map = Dict{Int, Int}(k => k for k in 1:10);
    vertex_input_dim::Union{Dict, Nothing} = nothing,
    continuous::Bool=false,
    kwargs...
  )
    g = NamedDiGraph(10) #             (1)
    add_edge!(g, 1, 6)   #              |
    add_edge!(g, 2, 6)   # (3)-(4)-(5)-(6)-(7)-(8)-(9)-[10]   
    add_edge!(g, 3, 4)   #              |
    add_edge!(g, 4, 5)   #             (2)
    add_edge!(g, 5, 6)
    add_edge!(g, 6, 7)
    add_edge!(g, 7, 8)
    add_edge!(g, 8, 9)
    add_edge!(g, 9, 10)
    vertex_input_dim = isnothing(vertex_input_dim) ? Dict(v => 2 for v in vertices(g)) : vertex_input_dim
    return continuous ? Structs.cTTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, kwargs...) :
           Structs.TTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, vertex_input_dim=vertex_input_dim, kwargs...)
  end

  function LinearWithTwoChildrenReverted(
    vertex_to_input_pos_map = Dict{Int, Int}(k => k for k in 1:10);
    vertex_input_dim::Union{Dict, Nothing} = nothing,
    continuous::Bool=false,
    kwargs...
  )
    g = NamedDiGraph(10) #              (9)
    add_edge!(g, 10, 5)  #               |
    add_edge!(g, 9, 5)   #  (8)-(7)-(6)-(5)-(4)-(3)-(2)-[1]
    add_edge!(g, 8, 7)   #               |
    add_edge!(g, 7, 6)   #              (10)
    add_edge!(g, 4, 3)
    add_edge!(g, 6, 5)
    add_edge!(g, 5, 4)
    add_edge!(g, 3, 2)
    add_edge!(g, 2, 1)
    vertex_input_dim = isnothing(vertex_input_dim) ? Dict(v => 2 for v in vertices(g)) : vertex_input_dim
    return continuous ? Structs.cTTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, kwargs...) :
           Structs.TTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, vertex_input_dim=vertex_input_dim, kwargs...)
  end

  function LargerExampleTree(
    vertex_to_input_pos_map = Dict{Int, Int}(k => k for k in 1:20);
    vertex_input_dim::Union{Dict, Nothing} = nothing,
    continuous::Bool=false,
    kwargs...
  )
    g = NamedDiGraph(20)
    add_edge!(g, 3, 1)
    add_edge!(g, 6, 3)
    add_edge!(g, 7, 6)
    add_edge!(g, 13, 7)
    add_edge!(g, 15, 13)
    add_edge!(g, 16, 15)
    add_edge!(g, 16, 15)
    add_edge!(g, 2, 1)
    add_edge!(g, 4, 2)
    add_edge!(g, 10, 4)
    add_edge!(g, 11, 4)
    add_edge!(g, 5, 2)
    add_edge!(g, 8, 5)
    add_edge!(g, 9, 5)
    add_edge!(g, 12, 9)
    add_edge!(g, 14, 12)
    add_edge!(g, 18, 14)
    add_edge!(g, 19, 14)
    add_edge!(g, 20, 18)
    vertex_input_dim = isnothing(vertex_input_dim) ? Dict(v => 2 for v in vertices(g)) : vertex_input_dim
    return continuous ? Structs.cTTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, kwargs...) :
           Structs.TTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, vertex_input_dim=vertex_input_dim, kwargs...)
  end

  function ExampleTreeFromThesis(
    vertex_to_input_pos_map = Dict{Int, Int}(k => k for k in 1:15);
    vertex_input_dim::Union{Dict, Nothing} = nothing,
    continuous::Bool=false,
    kwargs...
  )
    # Binary tree structure: root (1) has children 2,3
    # Node 2 has children 4,5; Node 3 has children 6,7
    # Nodes 4,5,6,7 each have 2 children (8-15)
    g = NamedDiGraph(15)
    # Root level: 2 and 3 point to 1
    add_edge!(g, 2, 1)
    add_edge!(g, 3, 1)
    # Second level: 4,5 point to 2; 6,7 point to 3
    add_edge!(g, 4, 2)
    add_edge!(g, 5, 2)
    add_edge!(g, 6, 3)
    add_edge!(g, 7, 3)
    # Third level: children of 4,5,6,7
    add_edge!(g, 8, 4)
    add_edge!(g, 9, 4)
    add_edge!(g, 10, 5)
    add_edge!(g, 11, 5)
    add_edge!(g, 12, 6)
    add_edge!(g, 13, 6)
    add_edge!(g, 14, 7)
    add_edge!(g, 15, 7)
    vertex_input_dim = isnothing(vertex_input_dim) ? Dict(v => 2 for v in vertices(g)) : vertex_input_dim
    return continuous ? Structs.cTTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, kwargs...) :
           Structs.TTNS(g; vertex_to_input_pos_map=vertex_to_input_pos_map, vertex_input_dim=vertex_input_dim, kwargs...)
  end
end
