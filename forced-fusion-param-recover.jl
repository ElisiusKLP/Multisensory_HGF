"""
Creating Forced-Fusion model with HGF.
Feeding with random (no patterns) simulated data/locations.

Trying to conduct parameter recovery.

"""
# Packages
using DataFrames, CSV, TimeSeries, Serialization
using Distributions
using Plots, StatsPlots
using ActionModels, HierarchicalGaussianFiltering
using Turing
using CategoricalArrays
using Distributed # for parallelization

# Remove workers (if needed)
#rmprocs(workers())

# Setting up multiple workeder for parralelization
## Set up multiple workers with HGF package
n_cores = 2
if n_cores > 1
    addprocs(n_cores, exeflags = "--project")
    @everywhere @eval using HierarchicalGaussianFiltering
    @everywhere @eval using ActionModels
    
end

#List of input nodes to create
input_nodes = [Dict(

    "name" => "A",
), Dict("name" => "V",)]

#List of state nodes to create
state_nodes = [
    Dict(
        "name" => "location",
    ),
]
#List of child-parent relations
edges = [
    Dict(
        "child" => "A",
        "value_parents" => ("location"),
    ),
    Dict(
        "child" => "V",
        "value_parents" => ("location"),
    ),
]
#Initialize the HGF
hgf = init_hgf(
    input_nodes = input_nodes,
    state_nodes = state_nodes,
    edges = edges,
)

get_parameters(hgf)

@everywhere function multisensory_hgf_action(agent::Agent, input)
    action_noise = agent.parameters["action_noise"]
    #Update hgf
    hgf = agent.substruct
    update_hgf!(hgf, input)
    #get out inferred location
    inferred_location = get_states(hgf, ("location", "posterior_mean"))
    #Create action distribution
    action_distribution = Normal(inferred_location, action_noise)
    return action_distribution
end

agent_parameters = Dict(
    "action_noise" => 1
)

agent = init_agent(
    multisensory_hgf_action,
    parameters = agent_parameters,
    substruct = hgf,
)

priors = Dict(
    ("V", "evolution_rate") => Normal(-2, 1),
    ("A", "evolution_rate") => Normal(-2, 1),
)


# Creating a simulation of five spatial locations similar to ventriliquist experiments
## Define the discrete values

values = [-22, -11, 0, 11, 22]
## Number of samples
num_samples = 1000
## Generate samples
# Generate samples of vectors consisting of two values
inputs = [rand(values, 2) for _ in 1:num_samples]
# To view the first few samples
println(inputs[1:10])

# Feeding the agent with inputs to generate some actions

# Parameter recovery

## create a list of two lists for each parameter which we want to test
# [-5, -2, 0, 2]
## i eventually want to create a list of all possible combinations of the parameters
## right now testing a 2x2 combination of parameters
A_er = [-5,-2]
V_er = [-5,-2]

# create a list of all possible combinations of the parameters
combinations = collect(Iterators.product(A_er, V_er))

# flatten the matrix of combinations into a list of n=2 vectors
param_combs = combinations[:]

# create an empty dicitonary to save the median parameters of each model fit
posterior_medians = Dict()


## TESTING FIT WITH parralelization

set_parameters!(agent, Dict(
        ("V", "evolution_rate") => -5,
        ("A", "evolution_rate") => 0,
    ))

reset!(agent)

give_inputs!(agent, inputs)

action_history = get_history(agent, "action")

action_history = action_history[2:1001]

result = fit_model(
        agent,
        priors,
        inputs,
        action_history,
        n_cores = 2,
        n_iterations = 1000,
        n_chains = 2,
    )
end

# With this method i can retrieve the median from all chains 
# and save them in a dictionary with the key being the parameter combination

get_posteriors(result[:,:,2], type = "median")

## i have not tested this but its a start
## require that n_chains is defined as an integer
## waiting to hear from peter
for chain in n_chains
    post_median = get_posteriors(result[:,:,chain], type = "median")

    post_median_V = post_median["V", "evolution_rate"]
    post_median_A = post_median["A", "evolution_rate"]

    if key in keys(posterior_medians)
        push!(posterior_medians[key][("V", "evolution_rate")], post_median_V)
        push!(posterior_medians[key][("A", "evolution_rate")], post_median_A)
    else
        posterior_medians[key] = Dict(
            ("V", "evolution_rate") => [post_median_V],
            ("A", "evolution_rate") => [post_median_A],
        )
    end
    z += n_chains
end


for i in param_combs

    v_rate = i[1]
    a_rate = i[2]

    set_parameters!(agent, Dict(
        ("V", "evolution_rate") => v_rate,
        ("A", "evolution_rate") => a_rate,
    ))

    key = string("V_",v_rate, "_A_", a_rate)

    z = 1

    while z <= 2 # number of iterations (could change to n if function)
        reset!(agent)

        give_inputs!(agent, inputs)

        action_history = get_history(agent, "action")

        action_history = action_history[2:1001]
        
        result = fit_model(
            agent,
            priors,
            inputs,
            action_history,
            n_cores = 2,
            n_iterations = 1000,
            n_chains = 2,
        )

        post_median = get_posteriors(result, type = "median")

        post_median_V = post_median["V", "evolution_rate"]
        post_median_A = post_median["A", "evolution_rate"]

        if key in keys(posterior_medians)
            push!(posterior_medians[key][("V", "evolution_rate")], post_median_V)
            push!(posterior_medians[key][("A", "evolution_rate")], post_median_A)
        else
            posterior_medians[key] = Dict(
                ("V", "evolution_rate") => [post_median_V],
                ("A", "evolution_rate") => [post_median_A],
            )
        end

        z += 1
    end
end
