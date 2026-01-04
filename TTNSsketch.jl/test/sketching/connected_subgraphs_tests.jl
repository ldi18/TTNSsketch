using Test
include("../../src/TTNSsketch.jl")
using .TTNSsketch

const SS = TTNSsketch.Sketching.SketchingSets
const ET = TTNSsketch.ExampleTopologies

@testset "connected_subgraphs_containing Linear(2)" begin
  ttns = ET.Linear(2)
  # Graph: 1 -- 2
  # k = 1: connected subsets S⊆{2} with S∪{1} connected -> {2}
  S1 = SS.connected_subgraphs_containing(ttns, 1)
  @test sort(S1) == [[2]]

  # k = 2: connected subsets S⊆{1} with S∪{2} connected -> {1}
  S2 = SS.connected_subgraphs_containing(ttns, 2)
  @test sort(S2) == [[1]]
end

@testset "connected_subgraphs_containing Linear(3)" begin
  ttns = ET.Linear(3)
  # Graph: 1 -- 2 -- 3

  # k = 1, S⊆{2,3} with {1}∪S connected:
  #   S={2}, {2,3} (but not {3} alone)
  S1 = sort(SS.connected_subgraphs_containing(ttns, 1))
  @test sort(S1) == sort([[2], [2,3]])

  # k = 2, S⊆{1,3} with {2}∪S connected:
  #   S={1}, {3}, {1,3}
  S2 = sort(SS.connected_subgraphs_containing(ttns, 2))
  @test sort(S2) == sort([[1], [3], [1,3]])

  # k = 3, symmetric to k=1
  S3 = sort(SS.connected_subgraphs_containing(ttns, 3))
  @test sort(S3) == sort([[2], [1,2]])
end

@testset "connected_subgraphs_containing ExampleTreeFromPaper" begin
  ttns = ET.ExampleTreeFromPaper()

  # Graph structure (undirected view):
  # 1-2-3, 2-4-5, 4-6-7, 7-8, 7-9, 7-10

  # Test a few hard-coded cases around the central node k=4.
  S4 = SS.connected_subgraphs_containing(ttns, 4)
  sets4 = Set(map(s -> Tuple(s), S4))

  # Single neighbors of 4 should be allowed.
  @test (2,) in sets4
  @test (5,) in sets4
  @test (6,) in sets4

  # Pairs of neighbors that share 4 as a hub are connected via 4,
  # but the induced subgraph on {4}∪S must be connected using only
  # edges among those vertices. For S={2,5}, path 2-4-5 exists, so
  # {2,5} is valid; similarly for {2,6} and {5,6}.
  @test (2,5) in sets4
  @test (2,6) in sets4
  @test (5,6) in sets4

  # A disconnected pair that would require intermediate vertices
  # not in S∪{4} should not appear. For S={1,7}, the induced
  # subgraph on {1,4,7} has no edges, so it is not connected.
  @test !((1,7) in sets4)

  # Also test a few cases for k=2.
  S2 = SS.connected_subgraphs_containing(ttns, 2)
  sets2 = Set(map(s -> Tuple(s), S2))
  @test (1,) in sets2       # 1-2
  @test (3,) in sets2       # 3-2
  @test (4,) in sets2       # 4-2
  @test (1,3) in sets2      # 1-2-3 connected via 2
  @test !((1,5) in sets2)   # 1 and 5 need 2-4 as intermediates; {1,2,5} has no 1-5 edge
end
