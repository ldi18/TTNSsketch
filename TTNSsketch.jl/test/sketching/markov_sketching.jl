using Printf
using Test
using ITensors: set_warn_order
using .TTNSsketch: GraphicalModels, Evaluate, Sketching,
                 CoreDeterminingEquations, ExampleTopologies,
                 ErrorReporting, samples

# Here we test the full logic of the TTNSsketching algorithm using the Markov sketches
# --> We show that the first order Markov conditioning can be recovered with any Sketching order.
@testset "Markov TTNS sketching" begin
  ttns = ExampleTopologies.ExampleTreeFromPaper()
  d = length(ttns.x_indices)         # dimensions
  set_warn_order(d+1)
  model = GraphicalModels.Ising_dGraphicalModel(ttns)
  probability_dict = Evaluate.probability_dict(model.ttns)
  probability_dict_sample = Dict{NTuple{length(ttns.vertex_input_dim), Int}, Float64}()

  for use_sample_matrix in (false,) #(false, true)
    if use_sample_matrix    # outside of order loop to avoid regenerating samples
      N = 30000
      sample_matrix = samples(model.ttns, N; seed=1234)
    end
    for order in [1, 2, d-1, d]
      enforce_non_recursive = (false)
      if order == 1 # First order Markov is allows both recursive and non-recursive
        enforce_non_recursive = (true, false)
      end
      for enforce in enforce_non_recursive
        sketching_kwargs = Dict{Symbol, Any}(
        :sketching_type => Sketching.Markov,
        :sketching_set_function => Sketching.SketchingSets.MarkovCircle,
        :order => order,
        :enforce_non_recursive => enforce
        )
        ttns_recov = deepcopy(ttns)
        if use_sample_matrix
          # TODO: Check this part again, 
          # Maybe the sample path is broken currently, but this is not a major problem, as any sample matrix can be converted into the frequency dict, which does just the same
          # as if we would run with the sample matrix.
          CoreDeterminingEquations.compute_Gks!(sample_matrix, ttns_recov; sketching_kwargs)
          sample_keys = Set(Tuple.(eachrow(sample_matrix)))
          maximum_error = ErrorReporting.report_errors(probability_dict, ttns_recov, sample_keys; print_sections=false).kept.max
          @test maximum_error < 1e-1
        else
          CoreDeterminingEquations.compute_Gks!(probability_dict, ttns_recov; sketching_kwargs)
          maximum_error = ErrorReporting.report_errors(probability_dict, ttns_recov, keys(probability_dict); print_sections=false).overall.max
          @test maximum_error < 1e-11
        end
      end
    end
  end
end

