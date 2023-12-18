"""
This is a parameter recovery function script for the Julia model.


"""

# Packages
using DataFrames, CSV, TimeSeries, Serialization
using Distributions
using Plots, StatsPlots
using ActionModels, HierarchicalGaussianFiltering
using Turing
using CategoricalArrays

function param_recover_test_sim(
    agent::Agent,
    inputs::Dict,
    priors::Array,
    n::Integer,
    A_er::Array,
    V_er::Array,
    n_cores::Integer = 1,
    n_iterations::Integer = 1000,
    n_chains = 2,
)

    # create a list of all possible combinations of the parameters
    combinations = collect(Iterators.product(A_er, V_er))

    # flatten the matrix of combinations into a list of n=2 vectors
    param_combs = combinations[:]

    # create an empty dicitonary to save the median parameters of each model fit
    posterior_medians = Dict()

    for i in param_combs

        v_rate = i[1]
        a_rate = i[2]

        set_parameters!(agent, Dict(
            ("V", "evolution_rate") => v_rate,
            ("A", "evolution_rate") => a_rate,
        ))

        key = string(v_rate, a_rate)

        z = 1

        while z <= n # number of iterations (could change to n if function)
            reset!(agent)

            give_inputs!(agent, inputs)

            action_history = get_history(agent, "action")

            action_history = action_history[2:1001]
            
            result = fit_model(
                agent,
                priors,
                inputs,
                action_history,
                n_cores,
                n_iterations,
                n_chains,
            )

            post_median = get_posteriors(result, type = "median")

            if key in keys(posterior_medians)
                push!(posterior_medians[key], post_median)
            else
                posterior_medians[key] = post_median
            end

            z += 1
        end
    end
end