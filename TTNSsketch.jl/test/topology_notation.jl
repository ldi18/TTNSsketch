using .TopologyNotation: C, R, N, L, R, P, N_depth_d, L_depth_d, R_depth_d, set_root!
using NamedGraphs: vertices, NamedDiGraph, is_directed
using NamedGraphs.GraphsExtensions: rem_edge!, add_edge!

@testset "Graph Hierarchy Labeling" begin
  # check type
  tree = ExampleTreeFromPaper().tree
  for v in vertices(tree)
    @test isa(C(tree, v), Vector{Int})
  end

  # check value C, P and N for all vertices
  @test C(tree, 1) == []
  @test C(tree, 2) == [1,3]
  @test C(tree, 3) == []
  @test C(tree, 4) == [2,5]
  @test C(tree, 5) == []
  @test C(tree, 6) == [4]
  @test C(tree, 7) == [6,8,9]
  @test C(tree, 8) == []
  @test C(tree, 9) == []
  @test C(tree, 10) == [7]

  @test P(tree, 1) == [2]
  @test P(tree, 2) == [4]
  @test P(tree, 3) == [2]
  @test P(tree, 4) == [6]
  @test P(tree, 5) == [4]
  @test P(tree, 6) == [7]
  @test P(tree, 7) == [10]
  @test P(tree, 8) == [7]
  @test P(tree, 9) == [7]
  @test P(tree, 10) == []

  @test N(tree, 1) == [2]
  @test N(tree, 2) == [1,3,4]
  @test N(tree, 3) == [2]
  @test N(tree, 4) == [2,5,6]
  @test N(tree, 5) == [4]
  @test N(tree, 6) == [4,7]
  @test N(tree, 7) == [6,8,9,10]
  @test N(tree, 8) == [7]
  @test N(tree, 9) == [7]
  @test N(tree, 10) == [7]

  # test for L
  @test L(tree, 4) == [1,2,3,5]
  @test L(tree, 7) == [1,2,3,4,5,6,8,9]
  @test L(tree, 1) == []
  @test L(tree, 10) == [1,2,3,4,5,6,7,8,9]

  # test for R
  @test R(tree, 10; ignore_siblings=false) == []
  @test R(tree, 4; ignore_siblings=false) == [6,7,8,9,10]
  @test R(tree, 1; ignore_siblings=false) == [2,3,4,5,6,7,8,9,10]
  @test R(tree, 7; ignore_siblings=false) == [10]
  @test R(tree, 5; ignore_siblings=false) == [1,2,3,4,6,7,8,9,10]

  @test R(tree, 10; ignore_siblings=true) == []
  @test R(tree, 4; ignore_siblings=true) == [6,7,8,9,10]
  @test R(tree, 1; ignore_siblings=true) == [2,4,5,6,7,8,9,10]
  @test R(tree, 7; ignore_siblings=true) == [10]
  @test R(tree, 5; ignore_siblings=true) == [4,6,7,8,9,10]

  # check value for N_depth_L 
  for v in vertices(tree)
    @test N_depth_d(tree, v, 1) == N(tree, v)   # trivial
  end
  @test N_depth_d(tree, 1, 2) == [2,3,4]
  @test N_depth_d(tree, 4, 2) == [1,2,3,5,6,7]
  @test N_depth_d(tree, 4, 3) == [1,2,3,5,6,7,8,9,10]
  @test N_depth_d(tree, 10, 4) == [2,4,5,6,7,8,9]

  # check L_depth_d
  @test L_depth_d(tree, 1, 1) == []
  @test L_depth_d(tree, 1, 2) == []
  @test L_depth_d(tree, 8, 2) == []
  @test L_depth_d(tree, 2, 1) == [1,3]
  @test L_depth_d(tree, 2, 5) == [1,3]
  @test L_depth_d(tree, 4, 2) == L(tree, 4)
  @test L_depth_d(tree, 4, 1) == [2,5]
  @test L_depth_d(tree, 6, 2) == [2,4,5]
  @test L_depth_d(tree, 7, 3) == [2,4,5,6,8,9]
  @test L_depth_d(tree, 7, 1) == [6,8,9]
  for v in vertices(tree)
    @test L_depth_d(tree, v, 9) == L(tree, v)   # trivial for 10 vertex graph
  end
  for v in vertices(tree)
    @test L_depth_d(tree, v, 0) == [] # test for depth 0
  end

  # check R_depth_d
  @test R_depth_d(tree, 1, 1; ignore_siblings=false) == [2]
  @test R_depth_d(tree, 1, 2; ignore_siblings=false) == [2,3,4]
  @test R_depth_d(tree, 2, 1; ignore_siblings=false) == [4]
  @test R_depth_d(tree, 2, 2; ignore_siblings=false) == [4,5,6]
  @test R_depth_d(tree, 2, 3; ignore_siblings=false) == [4,5,6,7]
  @test R_depth_d(tree, 2, 4; ignore_siblings=false) == [4,5,6,7,8,9,10]
  @test R_depth_d(tree, 3, 1; ignore_siblings=false) == [2]
  @test R_depth_d(tree, 3, 2; ignore_siblings=false) == [1,2,4]
  @test R_depth_d(tree, 3, 3; ignore_siblings=false) == [1,2,4,5,6]
  @test R_depth_d(tree, 3, 4; ignore_siblings=false) == [1,2,4,5,6,7]
  @test R_depth_d(tree, 3, 5; ignore_siblings=false) == [1,2,4,5,6,7,8,9,10]
  @test R_depth_d(tree, 3, 6; ignore_siblings=false) == [1,2,4,5,6,7,8,9,10]
  for v in vertices(tree)
    @test R_depth_d(tree, v, 9; ignore_siblings=false) == R(tree, v; ignore_siblings=false)   # trivial for 10 vertex graph
  end
  for v in vertices(tree)
    @test R_depth_d(tree, v, 0; ignore_siblings=false) == [] # test for depth 0
  end

  @test R_depth_d(tree, 1, 1; ignore_siblings=true) == [2]
  @test R_depth_d(tree, 1, 2; ignore_siblings=true) == [2,4]
  @test R_depth_d(tree, 2, 1; ignore_siblings=true) == [4]
  @test R_depth_d(tree, 2, 2; ignore_siblings=true) == [4,6]
  @test R_depth_d(tree, 2, 3; ignore_siblings=true) == [4,6,7]
  @test R_depth_d(tree, 2, 4; ignore_siblings=true) == [4,6,7,8,9,10]
  @test R_depth_d(tree, 3, 1; ignore_siblings=true) == [2]
  @test R_depth_d(tree, 3, 2; ignore_siblings=true) == [2,4]
  @test R_depth_d(tree, 3, 3; ignore_siblings=true) == [2,4,5,6]
  @test R_depth_d(tree, 3, 4; ignore_siblings=true) == [2,4,5,6,7]
  @test R_depth_d(tree, 3, 5; ignore_siblings=true) == [2,4,5,6,7,8,9,10]
  @test R_depth_d(tree, 3, 6; ignore_siblings=true) == [2,4,5,6,7,8,9,10]
  for v in vertices(tree)
    @test R_depth_d(tree, v, 9; ignore_siblings=true) == R(tree, v; ignore_siblings=true)   # trivial for 10 vertex graph
  end
  for v in vertices(tree)
    @test R_depth_d(tree, v, 0; ignore_siblings=true) == [] # test for depth 0
  end

  # consistency check for L_depth_d, R_depth_d and N_depth_d
  for v in vertices(tree)
    for depth in 1:9
      # check consistency for ignore_siblings=true
      siblings_and_their_descendants_depth_d = sort(!isempty(P(tree, v)) && (depth>1) ? vcat([union(s, L_depth_d(tree, s, depth-2)) for s in setdiff(C(tree, P(tree, v)[1]), v)]...) : [])
      @test isempty(intersect(siblings_and_their_descendants_depth_d, L_depth_d(tree, v, depth)))
      full_R_depth_d = sort(union(R_depth_d(tree, v, depth; ignore_siblings=true), siblings_and_their_descendants_depth_d))
      @test isempty(intersect(full_R_depth_d, L_depth_d(tree, v, depth)))
      @test L_depth_d(tree, v, depth) ∪ full_R_depth_d == N_depth_d(tree, v, depth)
    end
  end

  @testset "Setting the root" begin
    # Test set_root!
    tree = ExampleTreeFromPaper().tree
    set_root!(tree, 1)
    @test is_directed(tree)
  
    # The define the desired result for comparison
    test_tree = NamedDiGraph(10)
    add_edge!(test_tree, 2, 1)
    add_edge!(test_tree, 3, 2)
    add_edge!(test_tree, 4, 2)
    add_edge!(test_tree, 5, 4)
    add_edge!(test_tree, 6, 4)
    add_edge!(test_tree, 7, 6)
    add_edge!(test_tree, 8, 7)
    add_edge!(test_tree, 9, 7)
    add_edge!(test_tree, 10, 7)
  
    @test tree == test_tree
  end

end