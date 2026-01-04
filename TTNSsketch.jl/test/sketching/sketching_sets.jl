using Test
using .Sketching: SketchingSets
using .ExampleTopologies: ExampleTreeFromPaper

@testset "SketchingSets.MarkovCircle" begin
  ttns = ExampleTreeFromPaper()
  test_cases = Dict(
    1 => [
      (10, [7], Int[]),
      (7, [6, 8, 9], [10]),
      (4, [2, 5], [6]),
      (2, [1, 3], [4]),
      (1, Int[], [2]),
    ],
    2 => [
      (10, [6, 7, 8, 9], Int[]),
      (7, [4, 6, 8, 9], [10]),
      (6, [2, 4, 5], [7, 8, 9, 10]),
      (4, [1, 2, 3, 5], [6, 7]),
      (2, [1, 3], [4, 5, 6]),
    ],
    3 => [
      (10, [4, 6, 7, 8, 9], Int[]),
      (7, [2, 4, 5, 6, 8, 9], [10]),
      (6, [1, 2, 3, 4, 5], [7, 8, 9, 10]),
      (4, [1, 2, 3, 5], [6, 7, 8, 9, 10]),
      (3, Int[], [1, 2, 4, 5, 6]),
    ],
    # Edge case: depth larger than the tree diameter (capturing all descendants / non-descendants).
    11 => [
      (10, [1, 2, 3, 4, 5, 6, 7, 8, 9], Int[]),
      (7, [1, 2, 3, 4, 5, 6, 8, 9], [10]),
      (4, [1, 2, 3, 5], [6, 7, 8, 9, 10]),
      (1, Int[], [2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ]
  )

  for (depth, cases) in sort(collect(test_cases))
    sketching_kwargs = Dict(:order => depth)
    @testset "depth=$depth" begin
      for (vertex, expected_L, expected_R) in cases
        L_env, R_env = SketchingSets.MarkovCircle(ttns, vertex; sketching_kwargs=sketching_kwargs)
        @test L_env == expected_L
        @test R_env == expected_R
      end
    end
  end
end
