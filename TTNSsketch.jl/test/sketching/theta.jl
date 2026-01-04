include("../../src/TTNSsketch.jl")
using .TTNSsketch
using Random
using Test
using ITensors: ITensor, array

@testset "theta_ls recovers coefficients" begin
  rng = MersenneTwister(1234)
  ab_pairs = [(0.0, 1.0), (-1.0, 2.0), (0.5, 1.5)]

  for (a, b) in ab_pairs
  local_basis_kwargs = Dict{Int, Dict{Symbol, Any}}(
    1 => Dict{Symbol, Any}(
      :discrete => false,
      :local_basis_func_set => TTNSsketch.ContinuousVariableEmbedding.legendre_basis,
      :local_basis_func_kwargs => Dict{Symbol, Any}(:a => a, :b => b, :basis_expansion_order => 2)
    ),
    2 => Dict{Symbol, Any}(
      :discrete => false,
      :local_basis_func_set => TTNSsketch.ContinuousVariableEmbedding.legendre_basis,
      :local_basis_func_kwargs => Dict{Symbol, Any}(:a => a, :b => b, :basis_expansion_order => 3)
    ),
    3 => Dict{Symbol, Any}(
      :discrete => false,
      :local_basis_func_set => TTNSsketch.ContinuousVariableEmbedding.legendre_basis,
      :local_basis_func_kwargs => Dict{Symbol, Any}(:a => a, :b => b, :basis_expansion_order => 4)
    )
  )

    cttns = TTNSsketch.ExampleTopologies.Linear(3; continuous=true, local_basis_kwargs=local_basis_kwargs)
    dims = (cttns.vertex_input_dim[1], cttns.vertex_input_dim[2], cttns.vertex_input_dim[3])
    theta_true = randn(rng, dims...)

    basis_funcs = [local_basis_kwargs[i][:local_basis_func_set] for i in 1:3]
    basis_kwargs = [local_basis_kwargs[i][:local_basis_func_kwargs] for i in 1:3]

    function eval_f(x::Tuple{Float64, Float64, Float64})
      b1 = basis_funcs[1](x[1]; basis_kwargs[1]...)
      b2 = basis_funcs[2](x[2]; basis_kwargs[2]...)
      b3 = basis_funcs[3](x[3]; basis_kwargs[3]...)
      value = 0.0
      for i in 1:length(b1), j in 1:length(b2), k in 1:length(b3)
        value += theta_true[i, j, k] * b1[i] * b2[j] * b3[k]
      end
      return value
    end

    grid = range(a, b; length=8)
    f_dict = Dict{Tuple{Float64, Float64, Float64}, Float64}()
    for x1 in grid, x2 in grid, x3 in grid
      x = (Float64(x1), Float64(x2), Float64(x3))
      f_dict[x] = eval_f(x)
    end

    theta_hat = TTNSsketch.Sketching.ThetaSketchingFuncs.theta_ls(f_dict, [1, 2, 3], cttns)
    @test isapprox(theta_hat, theta_true; rtol=1e-8, atol=1e-8)
  end
end

@testset "theta_ls recovers coefficients with one discrete dimension" begin
  rng = MersenneTwister(4321)

  local_basis_kwargs = Dict{Int, Dict{Symbol, Any}}(
    1 => Dict{Symbol, Any}(
      :discrete => false,
      :local_basis_func_set => TTNSsketch.ContinuousVariableEmbedding.legendre_basis,
      :local_basis_func_kwargs => Dict{Symbol, Any}(:a => 0.0, :b => 1.0, :basis_expansion_order => 2)
    ),
    2 => Dict{Symbol, Any}(
      :discrete => false,
      :local_basis_func_set => TTNSsketch.ContinuousVariableEmbedding.legendre_basis,
      :local_basis_func_kwargs => Dict{Symbol, Any}(:a => 0.0, :b => 1.0, :basis_expansion_order => 3)
    ),
    3 => Dict{Symbol, Any}(
      :discrete => false,
      :local_basis_func_set => TTNSsketch.ContinuousVariableEmbedding.legendre_basis,
      :local_basis_func_kwargs => Dict{Symbol, Any}(:a => 0.0, :b => 1.0, :basis_expansion_order => 4)
    ),
    4 => Dict{Symbol, Any}(
      :discrete => true,
      :vertex_input_dim => 2
    )
  )

  cttns = TTNSsketch.ExampleTopologies.Linear(4; continuous=true, local_basis_kwargs=local_basis_kwargs)
  dims = (cttns.vertex_input_dim[1], cttns.vertex_input_dim[2], cttns.vertex_input_dim[3], cttns.vertex_input_dim[4])
  theta_true = 0.1 .* randn(rng, dims...)
  for d in 1:dims[4]
    theta_true[1, 1, 1, d] += 1.0
  end

  basis_funcs = [local_basis_kwargs[i][:local_basis_func_set] for i in 1:3]
  basis_kwargs = [local_basis_kwargs[i][:local_basis_func_kwargs] for i in 1:3]

  function eval_f(x::Tuple{Float64, Float64, Float64, Float64})
    b1 = basis_funcs[1](x[1]; basis_kwargs[1]...)
    b2 = basis_funcs[2](x[2]; basis_kwargs[2]...)
    b3 = basis_funcs[3](x[3]; basis_kwargs[3]...)
    d = Int(x[4])
    value = 0.0
    for i in 1:length(b1), j in 1:length(b2), k in 1:length(b3)
      value += theta_true[i, j, k, d] * b1[i] * b2[j] * b3[k]
    end
    return value
  end

  grid = range(0.0, 1.0; length=8)
  f_dict = Dict{Tuple{Float64, Float64, Float64, Float64}, Float64}()
  for d in 1:dims[4], x1 in grid, x2 in grid, x3 in grid
    x = (Float64(x1), Float64(x2), Float64(x3), Float64(d))
    f_dict[x] = eval_f(x)
  end

  vertex_selection = [1, 2, 3, 4]
  inds = (cttns.x_indices[1], cttns.x_indices[2], cttns.x_indices[3], cttns.x_indices[4])
  M_tensor = ITensor(inds...)
  TTNSsketch.Sketching.fill_marginal_distribution_tensor!(f_dict, M_tensor, vertex_selection, cttns)

  recovered = array(M_tensor, inds...)
  @test isapprox(recovered, theta_true; rtol=1e-8, atol=1e-8)
end

@testset "theta_ls recovers coefficients with two discrete dimensions" begin
  rng = MersenneTwister(9876)

  local_basis_kwargs = Dict{Int, Dict{Symbol, Any}}(
    1 => Dict{Symbol, Any}(
      :discrete => false,
      :local_basis_func_set => TTNSsketch.ContinuousVariableEmbedding.legendre_basis,
      :local_basis_func_kwargs => Dict{Symbol, Any}(:a => 0.0, :b => 1.0, :basis_expansion_order => 2)
    ),
    2 => Dict{Symbol, Any}(
      :discrete => false,
      :local_basis_func_set => TTNSsketch.ContinuousVariableEmbedding.legendre_basis,
      :local_basis_func_kwargs => Dict{Symbol, Any}(:a => 0.0, :b => 1.0, :basis_expansion_order => 3)
    ),
    3 => Dict{Symbol, Any}(
      :discrete => false,
      :local_basis_func_set => TTNSsketch.ContinuousVariableEmbedding.legendre_basis,
      :local_basis_func_kwargs => Dict{Symbol, Any}(:a => 0.0, :b => 1.0, :basis_expansion_order => 4)
    ),
    4 => Dict{Symbol, Any}(
      :discrete => true,
      :vertex_input_dim => 2
    ),
    5 => Dict{Symbol, Any}(
      :discrete => true,
      :vertex_input_dim => 3
    )
  )

  cttns = TTNSsketch.ExampleTopologies.Linear(5; continuous=true, local_basis_kwargs=local_basis_kwargs)
  dims = (
    cttns.vertex_input_dim[1],
    cttns.vertex_input_dim[2],
    cttns.vertex_input_dim[3],
    cttns.vertex_input_dim[4],
    cttns.vertex_input_dim[5]
  )
  theta_true = 0.1 .* randn(rng, dims...)
  for d4 in 1:dims[4], d5 in 1:dims[5]
    theta_true[1, 1, 1, d4, d5] += 1.0
  end

  basis_funcs = [local_basis_kwargs[i][:local_basis_func_set] for i in 1:3]
  basis_kwargs = [local_basis_kwargs[i][:local_basis_func_kwargs] for i in 1:3]

  function eval_f(x::Tuple{Float64, Float64, Float64, Float64, Float64})
    b1 = basis_funcs[1](x[1]; basis_kwargs[1]...)
    b2 = basis_funcs[2](x[2]; basis_kwargs[2]...)
    b3 = basis_funcs[3](x[3]; basis_kwargs[3]...)
    d4 = Int(x[4])
    d5 = Int(x[5])
    value = 0.0
    for i in eachindex(b1), j in eachindex(b2), k in eachindex(b3)
      value += theta_true[i, j, k, d4, d5] * b1[i] * b2[j] * b3[k]
    end
    return value
  end

  grid = range(0.0, 1.0; length=8)
  f_dict = Dict{Tuple{Float64, Float64, Float64, Float64, Float64}, Float64}()
  for d4 in 1:dims[4], d5 in 1:dims[5], x1 in grid, x2 in grid, x3 in grid
    x = (Float64(x1), Float64(x2), Float64(x3), Float64(d4), Float64(d5))
    f_dict[x] = eval_f(x)
  end

  vertex_selection = [1, 2, 3, 4, 5]
  inds = (
    cttns.x_indices[1],
    cttns.x_indices[2],
    cttns.x_indices[3],
    cttns.x_indices[4],
    cttns.x_indices[5]
  )
  M_tensor = ITensor(inds...)
  TTNSsketch.Sketching.fill_marginal_distribution_tensor!(f_dict, M_tensor, vertex_selection, cttns)

  recovered = array(M_tensor, inds...)
  @test isapprox(recovered, theta_true; rtol=1e-8, atol=1e-8)
end
