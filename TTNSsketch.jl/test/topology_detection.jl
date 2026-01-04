using Graphs: edges, src, dst, has_edge, Edge
using Random

using .TopologyDetection
using .ExampleTopologies
using .GraphicalModels

@testset "BMI Tests" begin
  # Create a dict with pairs as tuples, ranging from 1 to 5 respectively. Set a seed for a reproducible random number generation. Assign random probability to dict entries
  P = Dict{Tuple{Int, Int}, Float64}()
  Random.seed!(1234)
  for i in 1:5, j in 1:5
    P[(i, j)] = rand()
  end
  # Normalize the probabilities
  total_sum = sum(values(P))
  for key in keys(P)
    P[key] /= total_sum
  end
  bmi = BMI(P)
  @test isa(bmi, Number)
  @test bmi > 0
end

@testset "BMI Continuous/Mixed Tests" begin
  # Continous x, y
  x = range(0.0, 1.0; length=60)
  y = range(0.0, 1.0; length=60)

  Pxy_indep = Dict{Tuple{Float64, Float64}, Float64}()
  Pxy_dep = Dict{Tuple{Float64, Float64}, Float64}()

  for xi in x, yi in y
    Pxy_indep[(xi, yi)] = xi * yi       # No BMI for products
    Pxy_dep[(xi, yi)] = exp(xi * yi)    # Non-zero BMI, as in Graphical Model
  end
  bmi_indep = BMI(Pxy_indep)
  bmi_dep = BMI(Pxy_dep)
  @test abs(bmi_indep) < 1e-6
  @test bmi_dep > 1e-3

  # Mixed discrete/continuous case (discrete x, continuous y)
  x_disc = [0, 1]
  y_cont = range(0.0, 1.0; length=60)
  Pxy_mixed_indep = Dict{Tuple{Float64, Float64}, Float64}()
  Pxy_mixed_dep = Dict{Tuple{Float64, Float64}, Float64}()
  
  for xi in x_disc, yi in y_cont
    Pxy_mixed_indep[(xi, yi)] = (xi + 1.0) * yi
    Pxy_mixed_dep[(xi, yi)] = exp(xi * yi)
  end
  bmi_mixed_indep = BMI(Pxy_mixed_indep)
  bmi_mixed_dep = BMI(Pxy_mixed_dep)

  @test abs(bmi_mixed_indep) < 1e-6
  @test bmi_mixed_dep > 1e-3
end

@testset "Tree Topology Recovery Tree Graphical Model (Ising) - undirected" begin
  # Note: It is sufficient to check the recovery of undirected edges as the edge direction is only used for the hierarchy in the TTNS algorithm.
  model = GraphicalModels.Ising_dGraphicalModel(ExampleTreeFromPaper())

  # 1) Test recovery based on exact probabilities, i.e. a dict with the exact probability for every input
  P_exact_dict = Evaluate.probability_dict(model.ttns)
  spanning_tree_from_dict = maximum_spanning_tree_recovery(P_exact_dict)
  # Discard direction of edge, i.e. src and dst sorted
  sorted_edges_original = [Edge(min(src(e), dst(e)), max(src(e), dst(e))) for e in edges(model.ttns.tree)]
  sorted_edges_spanning_tree_from_dict = [Edge(min(src(e), dst(e)), max(src(e), dst(e))) for e in edges(spanning_tree_from_dict)]
  @test length(sorted_edges_original) == length(sorted_edges_spanning_tree_from_dict)
  @test all(e in sorted_edges_spanning_tree_from_dict for e in sorted_edges_original)
  @test all(e in sorted_edges_original for e in sorted_edges_spanning_tree_from_dict)

  # 2) Test recovery based on probabilities from samples, i.e. implied probabilities from samples (rows of matrix)
  P_implied_by_samples = samples(model.ttns, 10000; seed=1234)
  spanning_tree_from_samples = maximum_spanning_tree_recovery(P_implied_by_samples)
  sorted_edges_spanning_tree_from_samples = [Edge(min(src(e), dst(e)), max(src(e), dst(e))) for e in edges(spanning_tree_from_samples)]
  @test length(sorted_edges_original) == length(sorted_edges_spanning_tree_from_samples)
  @test all(e in sorted_edges_spanning_tree_from_samples for e in sorted_edges_original)
  @test all(e in sorted_edges_original for e in sorted_edges_spanning_tree_from_samples)
end
