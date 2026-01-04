# Markov sketching

module CoreDeterminingEquations
  using ..Preprocessing
  using ..TopologyNotation
  using ..Structs
  using ..Sketching: compute_Zk, compute_Zwk, Markov, Perturbative
  using ..Sketching.SketchingSets: MarkovCircle
  using ..Gauging: gauge!
  using Graphs
  using LinearAlgebra
  using ITensors
  import Statistics: mean

  export compute_Gks, compute_Gks!, contract_Gks

  """
  Intermediate term Ak from corrolary Eq. 17. Introduces indices alpha_{C(k), k}.
  """
  function compute_Ak(f::Union{Matrix{Int64}, Dict{<:Tuple, Float64}}, ttns::Structs.TTNSType{T}, k::T; sketching_kwargs) where {T}
    function compute_Awk(w::T)::ITensor
      # Recursive sketching (Cond. 4 satisfied, exploit Propos. 12)   
      if ((get(sketching_kwargs, :sketching_type, nothing) == Markov && get(sketching_kwargs, :order, 2) == 1) ||
         (get(sketching_kwargs, :sketching_type, nothing) == Perturbative)) && !get(sketching_kwargs, :enforce_non_recursive, false)
         Awk = sketching_kwargs[:sketching_type].compute_Awk_recursive(ttns, w; sketching_kwargs)
        return Awk
      end
      # General case
      Qw = ttns.Q[w]
      Zwk = compute_Zwk(f, ttns, w; sketching_kwargs)
      Awk = Zwk * Qw   # A_{w->k} = Z_{w->k} Q_w (Eq. 17, transpose implied by index contraction)
      return Awk
    end
    # Compute Ak = prod_w A_{w->k},
    # permute to (beta_{k, C(k)}, alpha_{k, C(k)}), with alpha and beta indices ordered by id, respectively
    Ak = prod([compute_Awk(w) for w in C(ttns.tree, k)])
    # Combine alpha indices if there are any
    alpha_inds_pos = findall(ind -> hastags(ind, "alpha"), inds(Ak))
    beta_inds_pos = findall(ind -> hastags(ind, "beta"), inds(Ak))
    perm = vcat(beta_inds_pos[sortperm(beta_inds_pos, by=x->id(inds(Ak)[x]))],
                alpha_inds_pos[sortperm(alpha_inds_pos, by=x->id(inds(Ak)[x]))])
    Ak = permute(Ak, inds(Ak)[perm]; allow_alias=true)
    ttns.alpha_children_combiner[k] = combiner([ind for ind in inds(Ak) if hastags(ind, "alpha")]; tags="alpha, C($(k))")
    Ak = ttns.alpha_children_combiner[k] * Ak
    isempty(beta_inds_pos) || (Ak = ttns.beta_inds_combiner[k] * Ak)
    return Ak   # (beta_{k, C(k)}, alpha_{k, C(k)})
  end

  """
  Computes Bk using (Eq. 8, 15). Introduces index alpha_{k, P(k)}.
  """
  function compute_Bk(f::Union{Matrix{Int64}, Dict{<:Tuple, Float64}}, ttns::Structs.TTNSType{T}, k; sketching_kwargs, svd_kwargs) where {T}
    Zk = compute_Zk(f, ttns, k; sketching_kwargs)
    if !isempty(P(ttns.tree, k))  # k is not the root
      svd_row_indices = [ind for ind in inds(Zk) if hastags(ind, "beta") || hastags(ind, "x")]
      Uk, Sk, Vk = svd(Zk, svd_row_indices...; lefttags="alpha,$(k), $(P(ttns.tree, k)[1])", svd_kwargs...)  # U-(alpha)-S-V
      Bk = Uk
      # If no recursive sketching can be used, store Qk for Ak computation
      if (!((get(sketching_kwargs, :sketching_type, nothing) == Markov && get(sketching_kwargs, :order, 2) == 1) ||
           (get(sketching_kwargs, :sketching_type, nothing) == Perturbative))) || get(sketching_kwargs, :enforce_non_recursive, false)
        # Invert Sk, using the same cutoff as the SVD if provided.
        local_cutoff = get(svd_kwargs, :cutoff, 1e-12)
        for i in eachindex(Sk)
          if abs(Sk[i]) > local_cutoff
            Sk[i] = 1 / Sk[i]
          end
        end
        ttns.Q[k] = Sk * Vk
      end
    else  # k is the root
      Bk = Zk
    end

    # Combine beta indices if there are any
    if !isempty(C(ttns.tree, k))
      beta_inds_pos = findall(ind -> hastags(ind, "beta"), inds(Bk))
      other_pos = findall(ind -> !hastags(ind, "beta"), inds(Bk))
      perm = vcat(beta_inds_pos[sortperm(beta_inds_pos, by=x->id(inds(Bk)[x]))], other_pos)

      Bk = permute(Bk, inds(Bk)[perm]; allow_alias=true)
      ttns.beta_inds_combiner[k] = combiner([ind for ind in inds(Bk) if hastags(ind, "beta")]; tags="beta,$(k),C($(k))")
      ttns.beta_inds_combined[k] = first(filter(ind -> hastags(ind, "beta,$(k),C($(k))"), inds(ttns.beta_inds_combiner[k])))  
      Bk = ttns.beta_inds_combiner[k] * Bk
      return Bk
    end
    ttns.beta_inds_combiner[k] = ITensor(1.0)
    return Bk   # Uk = Bk (Eq. 15)              # (beta_{k, C(k)}, x_k, alpha_{k, P(k)})
  end

  """
  Compute Gk for every vertex k. Uses Markov sketching functions.

  If `normalize_Gks=true`, the resulting TTNS is globally normalized by
  contracting all `Gk` cores to obtain the total mass Z, then scaling
  each core by `Z^(-1/|V|)` so that the overall network sums to 1.
  """
  function compute_Gks!(f::Union{Matrix{Int64}, Dict{<:Tuple, Float64}}, ttns::Structs.TTNSType{T};
                        sketching_kwargs::Dict{Symbol, Any},
                        svd_kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}(),
                        gauge_Gks::Bool = false,
                        unique_gauge::Bool = true,
                        normalize_Gks::Bool = false,
                        alpha_cutoff::Union{Nothing, Real}=nothing) where {T}
    # If an explicit alpha_cutoff is provided, forward it to the SVD as `cutoff`.
    if alpha_cutoff !== nothing
      svd_kwargs[:cutoff] = float(alpha_cutoff)
    end
    # Precompute Bks. This step includes the SVDs and produces results to be reused in the Ak computation.
    for k in vertices(ttns.tree)
      ttns.B[k] = compute_Bk(f, ttns, k; sketching_kwargs, svd_kwargs)
    end
    for k in vertices(ttns.tree)
      if isempty(C(ttns.tree, k))  # leaf
        ttns.G[k] = ttns.B[k]
      else
        Ak = compute_Ak(f, ttns, k; sketching_kwargs)
        beta_ind_pos = findfirst(ind -> hastags(ind, "beta"), inds(Ak))
        alpha_ind_pos = findfirst(ind -> hastags(ind, "alpha"), inds(Ak))
        beta_ind = inds(Ak)[beta_ind_pos]
        alpha_ind = inds(Ak)[alpha_ind_pos]

        Ak_pinv = pinv(Matrix(Ak, (beta_ind, alpha_ind)))
        Ak_pinv = ITensor(Ak_pinv, alpha_ind, beta_ind)
        ttns.G[k] = Ak_pinv * ttns.B[k]  # contract beta_C(k)
        decombine_alpha_children = dag(ttns.alpha_children_combiner[k])
        ttns.G[k] = decombine_alpha_children * ttns.G[k]
      end
    end
    if gauge_Gks
      gauge!(ttns; unique=unique_gauge)
    end

    if normalize_Gks
      Z = contract_Gks(ttns)
      # Distribute normalization evenly over all vertices
      scale = Z^(-1 / length(vertices(ttns.tree)))
      for k in vertices(ttns.tree)
        ttns.G[k] *= scale
      end
    end
  end

  """
  Contract all Gk cores into a scalar by multiplying them together and
  summing over all remaining x indices.
  """
  function contract_Gks(ttns::Structs.TTNSType{T}) where {T}
    # Multiply all Gk cores
    full = ITensor(1.0)
    for k in vertices(ttns.tree)
      full *= ttns.G[k]
    end
    # Sum over all remaining indices
    return sum(array(full))
  end

end
