using LinearAlgebra
using Random

using .ContinuousVariableEmbedding

"""
Simple orthogonality check for basis functions.
"""
function orthogonality_check(; a=0, b=1, basis_func_set=real_fourier_basis, num_samples=10000)
  X = a .+ (b - a) .* rand(num_samples)
  dim_basis_set = length(basis_func_set(0))
  B = Matrix{Float64}(undef, dim_basis_set, num_samples)
  B = hcat(basis_func_set.(X)...)
  G = B * B' .* ((b - a) / num_samples)
  return G # return Gram matrix
end

# G = orthogonality_check()
# println(isapprox(G, I, rtol=1e-1))

"""
Orthogonality check for product basis functions.
"""
function product_basis_orthogonality_check(n::Int, m::Int; basis_kwargs, num_samples)
  a = basis_kwargs[:local_basis_func_kwargs][:a]
  b = basis_kwargs[:local_basis_func_kwargs][:b]
  (X_samples, Y_samples) = ([a .+ (b - a) .* rand(n) for _ in 1:num_samples], [a .+ (b - a) .* rand(m) for _ in 1:num_samples])
  dim_basis_set = length(product_basis(X_samples[1], Y_samples[1]; X_kwargs=basis_kwargs, Y_kwargs=basis_kwargs))
  B = Matrix{Float64}(undef, dim_basis_set, num_samples)
  B = hcat(product_basis.(X_samples, Y_samples; X_kwargs=basis_kwargs, Y_kwargs=basis_kwargs)...)
  G = B * B' .* ((b - a)^(n+m) / num_samples)
  return G # return Gram matrix
end
  
@testset "Fourier/Legendre Basis Orthogonality Checks" begin
  # Set random period length T
  Random.seed!(1234)
  as = 5.0 * rand(5)
  bs = as .+ 2.0 * rand(5)

  for (a, b) in zip(as, bs)
    # Fourier Basis
    basis_kwargs = Dict{Symbol, Any}(
      :local_basis_func_set    => real_fourier_basis,
      :local_basis_func_kwargs => Dict{Symbol, Any}(:a => a, :b => b, :basis_expansion_order => 1)
    )
    G = product_basis_orthogonality_check(2, 3; basis_kwargs, num_samples=10000)
    @test maximum(abs.(G-I)) <= 0.1

    # Legendre Basis
    basis_kwargs = Dict{Symbol, Any}(
      :local_basis_func_set    => legendre_basis,
      :local_basis_func_kwargs => Dict{Symbol, Any}(:a => a, :b => b, :basis_expansion_order => 1)
    )
    G = product_basis_orthogonality_check(1, 1; basis_kwargs, num_samples=10000)
    @test maximum(abs.(G-I)) <= 0.1
  end

end