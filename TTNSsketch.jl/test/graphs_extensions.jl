using Test
using TTNSsketch
using NamedGraphs: vertices
using TTNSsketch.GraphsExtensions: edges_in_vertex_selection

@testset "GraphsExtensions.edges_in_vertex_selection" begin
  example = TTNSsketch.ExampleTopologies.ExampleTreeFromPaper()
  g = example.tree

  # Tests for vertex selections
  @test edges_in_vertex_selection(g, Int[]) == 0
  @test edges_in_vertex_selection(g, [1]) == 0
  @test edges_in_vertex_selection(g, [1, 2]) == 1
  @test edges_in_vertex_selection(g, [1, 4]) == 0
  @test edges_in_vertex_selection(g, [1, 2, 4]) == 2
  @test edges_in_vertex_selection(g, [2, 3]) == 1
  @test edges_in_vertex_selection(g, [5, 4, 6]) == 2
  @test edges_in_vertex_selection(g, [6, 7, 8, 9, 10]) == 4
  @test edges_in_vertex_selection(g, [1, 6, 7, 8, 9, 10]) == 4
  @test edges_in_vertex_selection(g, [1, 2, 6, 7, 8, 9, 10]) == 5
  @test edges_in_vertex_selection(g, [1, 2, 5, 7, 8, 9, 10]) == 4

  # Sanity check for the full vertex set
  d = length(vertices(g))
  @test edges_in_vertex_selection(g, collect(1:d)) == d - 1
end
