using ITensors: inds, hastags, delta, dag, combiner, replaceind!, dim, tags, Index, array
using .TopologyNotation: P
using .Evaluate: norm

using NamedGraphs: NamedEdge
using Graphs: edges


# TODO: Delete all the uniqueness related parts. Enforcing unique gauge is apparently not common to do.

@testset "Gauging Tests" begin
  local_basis_kwargs = Dict{Symbol, Any}(
    :local_basis_func_set => ContinuousVariableEmbedding.legendre_basis,
    :local_basis_func_kwargs => Dict{Symbol, Any}(:a => 0, :b => 1, :basis_expansion_order => 1)
  )

  # Define a random cTTNS
  cttns = ExampleTopologies.ExampleTreeFromPaper(; local_basis_kwargs, continuous=true)
  random_cttns = GraphicalModels.random_cGraphicalModel(cttns; seed=42, gauge_Gks=false).ttns

  # Define cTTNS with all coefficients being one
  cttns = ExampleTopologies.Linear(3; local_basis_kwargs, continuous=true)
  coefficients = Dict{NamedEdge{Int64}, AbstractArray}(
    edges(cttns.tree)[1] => ones(2, 2),
    edges(cttns.tree)[2] => ones(2, 2)
  )
  all1_cttns = GraphicalModels.GraphicalModel(cttns, coefficients; gauge_Gks=false).ttns

  for cttns in (random_cttns, all1_cttns)
    # Compute expectation value and contract tn before gauging
    norm_no_gauge = norm(cttns; exploit_gauge=false)
    contracted_tn_no_gauge = contract_ttns(cttns)
    contracted_tn_no_gauge = array(contracted_tn_no_gauge, sort(collect(inds(contracted_tn_no_gauge)), by=x->parse(Int, string(first(tags(x))))))

    for unique in (false, true)
      # Test if norm is the same with and without (unique) gauging
      gauged_cttns = deepcopy(cttns)
      Gauging.gauge!(gauged_cttns; unique=unique)
      norm_gauge = norm(gauged_cttns; exploit_gauge=true)
      contracted_tn_gauge = contract_ttns(gauged_cttns)
      contracted_tn_gauge = array(contracted_tn_gauge, sort(collect(inds(contracted_tn_gauge)), by=x->parse(Int, string(first(tags(x))))))
      @test isapprox(norm_no_gauge, norm_gauge)
      @test isapprox(contracted_tn_no_gauge, contracted_tn_gauge)
      
      # Check if the cores are isometries
      for (k, Gk) in gauged_cttns.G
        if !isempty(P(gauged_cttns.tree, k))  # Skip root
          Gk = copy(Gk)
          row_inds = filter(i -> hastags(i, "alpha,$(P(gauged_cttns.tree, k)[1])"), inds(Gk))
          row_ind_combiner = combiner(row_inds..., tags="rows")
          Gk = Gk * row_ind_combiner
          Gk_dag = copy(dag(Gk))
          row_ind = first(filter(i -> hastags(i, "rows"), inds(Gk)))
          row_ind_prime = Index(dim(row_ind), tags=tags(row_ind))
          replaceind!(Gk_dag, row_ind, row_ind_prime)
          GkGkdag = Gk * Gk_dag
          @test isapprox(GkGkdag, delta(row_ind, row_ind_prime))  # Test U * U^\dag = I
        end
      end
    end
  end

  # TODO: Find way to stabilize the uniqueness condition for all-one-matrix.
  # Check uniqueness of the gauge, if unique is set to true
  gauged_cttns = deepcopy(random_cttns)
  Gauging.gauge!(gauged_cttns; unique=true)
  G_previous = deepcopy(gauged_cttns.G)
  Gauging.gauge!(gauged_cttns; unique=true)  # Apply gauging on already gauged cTTNS
  for (k, Gk) in gauged_cttns.G
    inds_sorted_Gk_previous = sort(collect(inds(G_previous[k])), by=x -> join(tags(x)))
    inds_sorted_Gk = sort(collect(inds(Gk)), by=x -> join(tags(x)))
    @test isapprox(array(Gk, inds_sorted_Gk), array(G_previous[k], inds_sorted_Gk_previous))
  end
end