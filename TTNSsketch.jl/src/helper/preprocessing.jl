module Preprocessing
  using ..Structs
  export convert_sample_matrix_to_probability_dict

  """
  Convert a sample matrix into a probability dictionary.
  """
  function convert_sample_matrix_to_probability_dict(samples::Matrix{InputVariableType}, ttns::Structs.TTNSType{T}) where {InputVariableType, T}
    # Initialize the sample counts dictionary f with Float64 Tuple keys in the continious case and Int64 Tuple keys in the discrete case.
    # This is crucial because the function in Sketching use type overloading based on the key types.
    if isa(ttns, Structs.cTTNSType)
      return convert_sample_matrix_to_probability_dict(samples, Float64)
    elseif isa(ttns, Structs.dTTNSType)
      return convert_sample_matrix_to_probability_dict(samples, Int64)
    else
      error("TTNS type not recognized.")
    end
  end

  function convert_sample_matrix_to_probability_dict(samples::Matrix{InputVariableType}, key_type::Type{KeyType}) where {InputVariableType, KeyType}
    input_length = size(samples, 2)
    f = Dict{NTuple{input_length, KeyType}, Int}()
    for row in eachrow(samples)
      tuple_vec = Tuple(row)
      f[tuple_vec] = get(f, tuple_vec, 0) + 1
    end
    total_samples = sum(values(f))
    f = Dict(key => value / total_samples for (key, value) in f)
    return f
  end
end