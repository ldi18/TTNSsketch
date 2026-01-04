module TTNSsketch
  include("helper/properties.jl")
  include("structs.jl")
  include("helper/preprocessing.jl")
  include("ext/graphs_extensions.jl")
  include("ext/cross_interpolation.jl")
  include("topology_notation.jl")  
  include("topology_detection.jl")
  include("continuous_variable_embedding.jl")
  include("gauging.jl")
  include("evaluate.jl")
  include("graphical_models.jl")
  include("example_topologies.jl")
  include("sketching.jl")
  include("core_determining_equations.jl")
  include("helper/error_reporting.jl")

  using .Properties
  using .Structs
  using .TopologyDetection
  using .TopologyNotation
  using .Sketching
  using .CoreDeterminingEquations
  using .Evaluate
  using .ErrorReporting
  using .GraphicalModels
  using .ExampleTopologies
  using .ContinuousVariableEmbedding
  using .Gauging
  using .Preprocessing
  using .CrossInterpolation

  export Preprocessing
  isisometry = Properties.isisometry
  isunitary = Properties.isunitary
  export Properties, isisometry, isunitary
  TTNS = Structs.TTNS
  TTNSType = Structs.TTNSType
  dTTNSType = Structs.dTTNSType
  cTTNSType = Structs.cTTNSType
  export TTNS, TTNSType, dTTNSType, cTTNSType
  export Structs
  export TopologyDetection
  export TopologyNotation
  export Sketching
  export CoreDeterminingEquations
  export Evaluate
  export ErrorReporting
  evaluate = Evaluate.evaluate
  contract_ttns = Evaluate.contract_ttns
  norm = Evaluate.norm
  samples = Evaluate.samples
  export evaluate, contract_ttns, norm, samples
  export GraphicalModels
  export ExampleTopologies
  export ContinuousVariableEmbedding
  higher_order_probability_dict = GraphicalModels.higher_order_probability_dict
  export higher_order_probability_dict
  export Gauging
  gauge! = Gauging.gauge!
  export gauge!
  export CrossInterpolation

end
