module Properties
  using ITensors
  using LinearAlgebra: I, norm

  export isisometry, isunitary, Bks_column_space_overlap

  """
  Check if an ITensor A is an isometry when viewed as a matrix with row indices `row_inds`.
  The remaining indices are treated as column indices.
  """
  function isisometry(A::ITensor, row_inds::Union{Core.Array,Core.Tuple}; atol=1e-8)
    combiner_row_inds = combiner(row_inds; tags="row")
    combiner_col_inds = combiner(setdiff(inds(A), row_inds); tags="col")
    A = combiner_row_inds * A
    A = combiner_col_inds * A
    row_ind = first(filter(ind -> hastags(ind, "row"), inds(A)))
    col_ind = first(filter(ind -> hastags(ind, "col"), inds(A)))
    A = matrix(A, row_ind, col_ind)
    id = Matrix{eltype(A)}(I, size(A, 2), size(A, 2))
    return norm(A' * A - id) ≤ atol
  end

  """
  Check if an ITensor A is unitary when viewed as a matrix with row indices `row_inds`.
  The remaining indices are treated as column indices.
  """
  function isunitary(A::ITensor, row_inds::Union{Core.Array,Core.Tuple}; atol=1e-8)
    combiner_row_inds = combiner(row_inds; tags="row")
    combiner_col_inds = combiner(setdiff(inds(A), row_inds); tags="col")
    A = combiner_row_inds * A
    A = combiner_col_inds * A
    row_ind = first(filter(ind -> hastags(ind, "row"), inds(A)))
    col_ind = first(filter(ind -> hastags(ind, "col"), inds(A)))
    A = matrix(A, row_ind, col_ind)
    if size(A, 1) != size(A, 2)
      println("Matrix of size $(size(A)) is not square, cannot be unitary.")
      return false
    end
    id = Matrix{eltype(A)}(I, size(A, 1), size(A, 2))
    return norm(A' * A - id) ≤ atol
  end

  """
  Interprets Bk as a matrix with row indices x and column indices (beta, alpha). Uses SVD to obtain an orthonormal basis for col(Bk).
  Compares the column spaces of Bk1 and Bk2 via the singular values of the cross-gram matrix.
  """
  function Bks_column_space_overlap(Bk1::ITensor, Bk2::ITensor)
    # beta and x indices are row indices, alpha indices are column indices
    row_inds_1 = filter(i -> hastags(i, "x"), inds(Bk1))
    row_inds_2 = filter(i -> hastags(i, "x"), inds(Bk2))
    UBk1 = svd(Bk1, row_inds_1...).U
    UBk2 = svd(Bk2, row_inds_2...).U
    col_inds_1 = setdiff(inds(UBk1), row_inds_1)
    col_inds_2 = setdiff(inds(UBk2), row_inds_2)
    @assert isisometry(UBk1, row_inds_1; atol=1e-8) "Bk1 is not an isometry"
    @assert isisometry(UBk2, row_inds_2; atol=1e-8) "Bk2 is not an isometry"
    UBk1 = array(UBk1, row_inds_1..., col_inds_1...)
    UBk2 = array(UBk2, row_inds_2..., col_inds_2...)
    UBk1 = reshape(UBk1, (prod(dim.(row_inds_1)), prod(dim.(col_inds_1))))
    UBk2 = reshape(UBk2, (prod(dim.(row_inds_2)), prod(dim.(col_inds_2))))
    G = UBk1' * UBk2
    return svd(G).S
  end

end