using Test
using TTNSsketch
using TTNSsketch.CrossInterpolation
using LinearAlgebra
using ITensors

@testset "Cross Interpolation" begin
  @testset "Matrix operations" begin
    # Test eye function
    I3 = CrossInterpolation.eye(3, 3)
    @test size(I3) == (3, 3)
    @test I3[1, 1] == 1.0
    @test I3[2, 2] == 1.0
    @test I3[3, 3] == 1.0
    @test I3[1, 2] == 0.0
    @test I3[2, 1] == 0.0

    # Test backsolveL
    L = [1.0 0.0 0.0;
         0.5 1.0 0.0;
         0.3 0.2 1.0]
    iL = CrossInterpolation.backsolveL(L)
    @test size(iL) == (3, 3)
    @test isapprox(L * iL, Matrix{Float64}(I, 3, 3), atol=1e-10)

    # Test backsolveU
    U = [1.0 0.4 0.6;
         0.0 1.0 0.3;
         0.0 0.0 1.0]
    iU = CrossInterpolation.backsolveU(U)
    @test size(iU) == (3, 3)
    @test isapprox(U * iU, Matrix{Float64}(I, 3, 3), atol=1e-10)
  end

  @testset "prrldu decomposition" begin
    # Test with a simple matrix
    M = [1.0 2.0 3.0;
         4.0 5.0 6.0;
         7.0 8.0 9.0]
    
    L, d, U, pr, pc, inf_error = CrossInterpolation.prrldu(M; cutoff=0.0, maxdim=3)
    
    # Check dimensions
    @test size(L) == (3, length(d))
    @test length(d) <= 3
    @test size(U) == (length(d), 3)
    
    # Check reconstruction
    M_recon = L[pr, :] * Diagonal(d) * U[:, pc]
    @test isapprox(M_recon, M, atol=1e-10)
    
    # Test with cutoff
    L2, d2, U2, pr2, pc2, inf_error2 = CrossInterpolation.prrldu(M; cutoff=1e-6, maxdim=3)
    @test length(d2) <= length(d)
    
    # Test with maxdim
    L3, d3, U3, pr3, pc3, inf_error3 = CrossInterpolation.prrldu(M; cutoff=0.0, maxdim=2)
    @test length(d3) <= 2
  end

  @testset "Matrix interpolative decomposition" begin
    # Test with a simple matrix
    M = [1.0 2.0 3.0 4.0;
         5.0 6.0 7.0 8.0;
         9.0 10.0 11.0 12.0]
    
    C, Z, piv_cols, inf_error = CrossInterpolation.interpolative(M; cutoff=0.0, maxdim=4)
    
    # Check dimensions
    @test size(C) == (3, length(piv_cols))
    @test size(Z) == (length(piv_cols), 4)
    @test length(piv_cols) <= 3
    
    # Check reconstruction
    M_recon = C * Z
    @test isapprox(M_recon, M, atol=1e-10)
    
    # Check that C consists of columns of M
    for (i, col_idx) in enumerate(piv_cols)
      @test isapprox(C[:, i], M[:, col_idx], atol=1e-10)
    end
    
    # Test with cutoff
    C2, Z2, piv_cols2, inf_error2 = CrossInterpolation.interpolative(M; cutoff=1e-6, maxdim=4)
    @test length(piv_cols2) <= length(piv_cols)
    
    # Test with maxdim
    C3, Z3, piv_cols3, inf_error3 = CrossInterpolation.interpolative(M; cutoff=0.0, maxdim=2)
    @test length(piv_cols3) <= 2
  end

  @testset "ITensor interpolative decomposition" begin
    # Create a simple ITensor
    i1 = Index(3, "i1")
    i2 = Index(4, "i2")
    i3 = Index(2, "i3")
    
    T = randomITensor(i1, i2, i3)
    
    # Test interpolative decomposition
    col_inds = [i2, i3]
    C, Z, inf_error = CrossInterpolation.interpolative(T, col_inds; cutoff=0.0, maxdim=8, tags="Link")
    
    # Check that C and Z have correct indices
    # C should have row indices (i1) and link index
    # Z should have link index and column indices (i2, i3)
    @test i1 in inds(C)
    link_ind_C = filter(ind -> hastags(ind, "Link"), inds(C))
    @test !isempty(link_ind_C)
    # Z should have the column indices (combined) and link index
    link_ind_Z = filter(ind -> hastags(ind, "Link"), inds(Z))
    @test !isempty(link_ind_Z)
    
    # Check that there's a connecting index
    link_ind_C = filter(ind -> hastags(ind, "Link"), inds(C))
    link_ind_Z = filter(ind -> hastags(ind, "Link"), inds(Z))
    @test length(link_ind_C) == 1
    @test length(link_ind_Z) == 1
    @test link_ind_C[1] == link_ind_Z[1]
    
    # Check reconstruction
    T_recon = C * Z
    @test isapprox(array(T), array(T_recon), atol=1e-10)
    
    # Test with cutoff
    C2, Z2, inf_error2 = CrossInterpolation.interpolative(T, col_inds; cutoff=1e-6, maxdim=8, tags="Link")
    @test dim(link_ind_C[1]) >= dim(filter(ind -> hastags(ind, "Link"), inds(C2))[1])
  end

  @testset "Comparison with SVD in compute_Bk" begin
    # Create a simple TTNS
    using TTNSsketch.ExampleTopologies
    using TTNSsketch.CoreDeterminingEquations
    
    ttns = ExampleTopologies.Linear(3; vertex_input_dim=Dict(1 => 2, 2 => 2, 3 => 2))
    
    # Create a simple probability dictionary
    prob_dict = Dict(
      (1, 1, 1) => 0.1,
      (1, 1, 2) => 0.2,
      (1, 2, 1) => 0.15,
      (1, 2, 2) => 0.15,
      (2, 1, 1) => 0.1,
      (2, 1, 2) => 0.1,
      (2, 2, 1) => 0.1,
      (2, 2, 2) => 0.1
    )
    
    sketching_kwargs = Dict(
      :sketching_type => TTNSsketch.Sketching.Markov,
      :sketching_set_function => TTNSsketch.Sketching.SketchingSets.MarkovCircle,
      :order => 1
    )
    
    # Test with SVD (default)
    ttns_svd = deepcopy(ttns)
    svd_kwargs = Dict(:cutoff => 1e-12)
    Bk_svd = CoreDeterminingEquations.compute_Bk(prob_dict, ttns_svd, 2; sketching_kwargs, svd_kwargs)
    
    # Test with cross interpolation
    ttns_ci = deepcopy(ttns)
    ci_kwargs = Dict(:cutoff => 1e-12, :use_cross_interpolation => true)
    Bk_ci = CoreDeterminingEquations.compute_Bk(prob_dict, ttns_ci, 2; sketching_kwargs, svd_kwargs=ci_kwargs)
    
    # Both should produce valid Bk tensors with alpha indices
    alpha_inds_svd = filter(ind -> hastags(ind, "alpha"), inds(Bk_svd))
    alpha_inds_ci = filter(ind -> hastags(ind, "alpha"), inds(Bk_ci))
    @test !isempty(alpha_inds_svd)
    @test !isempty(alpha_inds_ci)
    
    # The dimensions might differ slightly, but both should be reasonable
    alpha_svd = filter(ind -> hastags(ind, "alpha"), inds(Bk_svd))[1]
    alpha_ci = filter(ind -> hastags(ind, "alpha"), inds(Bk_ci))[1]
    @test dim(alpha_svd) > 0
    @test dim(alpha_ci) > 0
  end

  @testset "Full workflow with cross interpolation" begin
    # Create a simple TTNS
    using TTNSsketch.ExampleTopologies
    using TTNSsketch.CoreDeterminingEquations
    
    ttns = ExampleTopologies.Linear(3; vertex_input_dim=Dict(1 => 2, 2 => 2, 3 => 2))
    
    # Create a simple probability dictionary
    prob_dict = Dict(
      (1, 1, 1) => 0.1,
      (1, 1, 2) => 0.2,
      (1, 2, 1) => 0.15,
      (1, 2, 2) => 0.15,
      (2, 1, 1) => 0.1,
      (2, 1, 2) => 0.1,
      (2, 2, 1) => 0.1,
      (2, 2, 2) => 0.1
    )
    
    sketching_kwargs = Dict(
      :sketching_type => TTNSsketch.Sketching.Markov,
      :sketching_set_function => TTNSsketch.Sketching.SketchingSets.MarkovCircle,
      :order => 1
    )
    
    # Test full compute_Gks! with cross interpolation
    ttns_ci = deepcopy(ttns)
    svd_kwargs = Dict{Symbol, Any}(:cutoff => 1e-12, :use_cross_interpolation => true)
    
    CoreDeterminingEquations.compute_Gks!(
      prob_dict, ttns_ci;
      sketching_kwargs=sketching_kwargs,
      svd_kwargs=svd_kwargs,
      normalize_Gks=false
    )
    
    # Check that G cores were computed
    @test haskey(ttns_ci.G, 1)
    @test haskey(ttns_ci.G, 2)
    @test haskey(ttns_ci.G, 3)
    
    # Check that we can evaluate the TTNS
    val1 = TTNSsketch.evaluate(ttns_ci, (1, 1, 1))
    val2 = TTNSsketch.evaluate(ttns_ci, (2, 2, 2))
    @test isfinite(val1)
    @test isfinite(val2)
  end
end

