"""
Try to run BCI mode_fit
"""


# Packages
using DataFrames, CSV, TimeSeries, Serialization
using Distributions
using Plots, StatsPlots
using ActionModels, HierarchicalGaussianFiltering
using Turing
using CategoricalArrays
using Distributed # for parallelization


"""
Creating the BCI in action models framework
without using a HGF substruct 
"""

#Common, independent
# prior for common: P_common

#Hvis common: location S_AV
# Gaussian(mean_AV, sigP) 


#Parameters: 
# p_common - prior for Common or not
# muP - centrality bias
# sigP - position variability from 0
# sigA - auditory noise
# sigV - visual noise

#States:
# C - whether common or not
# S_AV - the shared position
# S_A - the auditory position
# S_V - the visual position

#observations
# xA
# xV

# loading in "original_action_model"
include("$(pwd())/action_functions/bci_action.jl")

# If i wanted to save the posterior_C 
#agent.states["C"] = posterior_C
#push!(agent.history["C"], posterior_C)

original_params = Dict(
    #PArametrs: 
    "p_common" => 0.5 ,# p_common - prior for Common or not
    "muP" => 0,# muP - centrality bias
    "sigP" => 1,# sigP - position variability from 0
    "sigA" => 1,# sigA - auditory noise
    "sigV" => 1,# sigV - visual noise
    "action_noise" => 1
)

#States:
# C - whether common or not
# S_AV - the shared position
# S_A - the auditory position
# S_V - the visual position

original_states = [
    Dict("name" => "C"),
    Dict("name" => "sAV"),
    Dict("name" => "sA"),
    Dict("name" => "sV"),
]

original_states = Dict(
    "C" => 1,
    "sAV" => 0,
    "sA" => 0,
    "sV" => 0,
)

priors = Dict(
    "p_common" => Normal(0,1),
    "sigP" => lognormal(0, 1),
    "sigA" => Normal(0, 1),
    "sigV" => Normal(0, 1),
    "action_noise" => LogNormal(0,1),
)
# prøv at fit uden muP
# og smallere priors


agent = init_agent(
    original_action_model,
    parameters = original_params,
    states = original_states
)

get_parameters(agent)

# SIMULATING DATA
values = [-22, -11, 0, 11, 22]
## Number of samples
num_samples = 1000
## Generate samples
actions = Array(rand(values, num_samples))
## To view the first few samples
println(actions[1:10])
# Generate samples of vectors consisting of two values
inputs = Array( [rand(values, 2) for _ in 1:num_samples] )
# To view the first few samples
println(inputs[1:10])

get_parameters(agent)

give_inputs!(agent, inputs)
get_history(agent)
plot_trajectory(agent, "C") 
plot_trajectory(agent, "action")

inputs

# FITTING REAL DATA

dataset = CSV.read("park_and_kayser2023.csv", DataFrame)

df_exp1 = dataset[dataset[!, "experiment"] .== "experiment 1", :]

# Fitting independent group models

input_cols = [:auditory_location, :visual_location]
action_cols = [:action]
independent_group_cols = [:experiment, :subject]

chains = fit_model(
    agent,
    priors,
    df_exp1;
    input_cols = input_cols,
    action_cols = action_cols,
    independent_group_cols = independent_group_cols,
    n_iterations = 1000,
    n_cores = 1,
    n_chains = 2,
)

agent.history

# Fitting the model to the simulated data
chains = fit_model(
    agent,
    priors,
    inputs_exp1,
    actions_exp1,
    n_iterations = 2000,
)

write("bci_fit2_21-12-23.jls", chains)

part = chains[String15("experiment 1"), String31("participant 1.15")]

plot(part)

plot_parameter_distribution(part, priors)

plot_predictive_simulation(
    part,
    agent,
    inputs_exp1,
    ("action");
    n_simulations = 3
)

get_posteriors(part)

#----------------
##NOTES
#Fitting group level models

input_cols = [:auditory_location, :visual_location]
action_cols = [:action]
independent_group_cols = [:experiment, :participant]

chains = fit_model(
    agent,
    priors,
    dataset;
    input_cols = input_cols,
    action_cols = action_cols,
    independent_group_cols = independent_group_cols,
    n_iterations = 1000,
    n_cores = 4,
    n_chains = 2,
)






input_cols = [:auditory_input, :visual_input]
action_cols = [:action]
independent_group_cols = [:experiment]
multilevel_cols = [:participant]

priors = Dict(
    ("xA", "input_noise") => Multilevel(
        :participant,
        LogNormal,
        ["xA_noise_group_mean", "xA_noise_group_sd"]
    ),
    "xA_noise_group_mean" => Normal(0, 1),
    "xA_noise_group_sd" => LogNormal(0, 1),
)

chains = fit_model(
    agent,
    priors,
    dataset;
    input_cols = input_cols,
    action_cols = action_cols,
    independent_group_cols = independent_group_cols,
    multilevel_cols = multilevel_cols,
    n_iterations = 1000,
    n_cores = 4,
    n_chains = 2,
)


------
"This is for the merging HGF"




forced_fusionc
# FF_S_V
#FF_S_A

Independet
 #IND_S_V



model_comp




edges, nodes



init_hgf