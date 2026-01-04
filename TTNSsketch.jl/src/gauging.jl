module Gauging

  using ..Structs: TTNSType
  using ..TopologyNotation
  using ITensors
  using Graphs: indegree, outdegree, vertices

  export gauge!, gauge

  """
  A function to gauge a (c)TTNS with respect to root vertex, defined in CTTNS.tree.

  The unique option is not working reliably yet.
  """
  function gauge!(ttns::TTNSType{T}; unique::Bool=true)::TTNSType where {T}
    if length(keys(ttns.G)) == 0
      throw(ArgumentError("The TTNS must have initialized cores Gk to be gauged."))
    end
    # Idea: SVD-decompose all vertices with (equal) maximum distrance from the root vertex.
    # Then move SV to the parent vertex and repeat until the root vertex is reached. In every step
    # the vertices we decompose have identical distance from the root vertex. 
    
    # Iterate over list [[v with max depth...], [v with max depth - 1], ..., [children of root]]
    for level in depth_sorted_vertex_iterator(ttns.tree; include_root=false)
      for k in level
        p = P(ttns.tree, k)[1]
        # Only parent inds as column ind. This is the direction we want to move SV^\dag to.
        ind_to_parent = first(filter(i -> hastags(i, "alpha,$(p)"), inds(ttns.G[k])))
        row_inds = setdiff(inds(ttns.G[k]), [ind_to_parent])
        U, S, V = svd(ttns.G[k], row_inds..., lefttags = tags(ind_to_parent))
        Q = (S * V)
        # To ensure a unique gauge (accending sorting of singular values is not unique for unitary matrices - all singular values are 1)
        # sort by the values in the first row of U and fix the sign such that the first row is non-negative.
        if unique
          SVD_col_ind = first(filter(i -> hastags(i, "alpha,$(p)"), inds(U)))
          U_mat = array(U, row_inds..., SVD_col_ind)
          U_mat = reshape(U_mat, (prod(dim.(row_inds)), dim(SVD_col_ind)))
          U_mat = round.(U_mat; digits=16)
          signs_first_row = [abs(val) > 1e-15 ? sign(val) : 1 for val in first(eachrow(U_mat))] # Pin the sign of the first row
          U_mat = U_mat .* signs_first_row'
          #first_row = [U[[row_ind => 1 for row_ind in row_inds]..., SVD_alpha => SVD_alpha_val] for SVD_alpha_val in eachval(SVD_alpha)]
          #signs_first_row = [abs(val) > 1e-16 ? sign(val) : 1 for val in first_row]
          #sorting_permutation = sortperm(first_row .* signs_first_row; rev=true)
          sorting_permutation = sortperm(eachcol(U_mat); rev=true)  # Find a unique order of the columns
          SVD_col_ind_prime = Index(dim(SVD_col_ind), tags=tags(SVD_col_ind)) # The new column index after permutation
          perm_tensor = ITensor(SVD_col_ind_prime, SVD_col_ind)
          for i in eachval(SVD_col_ind)
            perm_tensor[SVD_col_ind => i, SVD_col_ind_prime => sorting_permutation[i]] = signs_first_row[i]
          end
          U = U * perm_tensor
          Q = Q * perm_tensor
        end
        ttns.G[k] = U
        ttns.G[p] = ttns.G[p] * Q
      end
    end
    return ttns
  end

  function gauge(ttns::TTNSType{T})::TTNSType where {T}
    ttns_copy = deepcopy(ttns)
    return gauge!(ttns_copy)
  end
end