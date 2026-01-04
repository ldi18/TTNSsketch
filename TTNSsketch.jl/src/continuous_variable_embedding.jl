module ContinuousVariableEmbedding
  using LinearAlgebra, Random
  using Statistics: mean
  using LegendrePolynomials
  
  export real_fourier_basis,
         legendre_basis,
         product_basis

  """
  Returns the real Fourier basis functtions [1, cos(2πx/T), sin(2πx/T), ..., cos(2πKx/T), sin(2πKx/T)]
  on a intervall [a, b], and number of harmonics K, evaluated at x.
  """
  function real_fourier_basis(x; a=0, b=1, basis_expansion_order=3)
    T = b - a
    ks = 1:basis_expansion_order
    cos_basis_functions = sqrt(2 / T) * cos.(ks * (x - a) * 2π / T)
    sin_basis_functions = sqrt(2 / T) * sin.(ks * (x - a) * 2π / T)
    return vcat(sqrt(1 / T), cos_basis_functions, sin_basis_functions)
  end

  """
  Returns the first K Legendre polynomials evaluated at x, with intervall [-1, 1] transformed to [a, b]
  """
  function legendre_basis(x; a=0, b=1, basis_expansion_order=3)
    T = b - a
    xi = 2 * (x - a) / T - 1
    normalization_factors = sqrt.((2 * (0:basis_expansion_order) .+ 1) / T)
    return normalization_factors .* parent(collectPl(xi, lmax = basis_expansion_order))
  end

  """
  Input: X = [x1, x2, ..., xd1], Y = [y1, y2, ..., yd2]
  Returns the tensor product b_x1 ⊗ b_x2 ⊗ ... ⊗ b_xd1 ⊗ b_y1 ⊗ b_y2 ⊗ ... ⊗ b_yd2 where
  each b is the vector of basis functions evaluated at the corresponding point.
  """
  function product_basis(X::AbstractArray, Y::AbstractArray; X_kwargs, Y_kwargs)
    X_basis_evaluated = map(x -> X_kwargs[:local_basis_func_set](x; X_kwargs[:local_basis_func_kwargs]...), X)
    Y_basis_evaluated = map(y -> Y_kwargs[:local_basis_func_set](y; Y_kwargs[:local_basis_func_kwargs]...), Y)
    return reduce(kron, reverse(vcat(X_basis_evaluated, Y_basis_evaluated)))
    # reverse added to simplify column-major reshaping into tensor (k1, k2, ..., kd)
  end
    
  """
  Allow single variables X = x1, Y = y1 as input.
  """
  function product_basis(x::Number, y::Number; X_kwargs, Y_kwargs)
    return product_basis([x], [y]; X_kwargs=X_kwargs, Y_kwargs=Y_kwargs)
  end

end
