"""
Merging HGF 

"""


# Packages
using DataFrames, CSV, TimeSeries, Serialization, JSON
using Distributions
using Plots, StatsPlots
using ActionModels, HierarchicalGaussianFiltering
using Turing
using CategoricalArrays
using ForwardDiff
using NNlib

"This is for the merging HGF"

#List of input nodes to create
input_nodes = [
    Dict("name" => "FF_A"),
    Dict("name" => "FF_V"),
    Dict("name" => "Seg_A"),
    Dict("name" => "Seg_V"),
]


#List of state nodes to create
state_nodes = [
    Dict("name" => "FF_sAV"),
    Dict("name" => "Seg_sA"),
    Dict("name" => "Seg_sV"),
]

edges = [
    # Forced fusion
    Dict("child" => "FF_A", "value_parents" => ("FF_sAV")),
    Dict("child" => "FF_V", "value_parents" => ("FF_sAV")),
    # Independent
    Dict("child" => "Seg_A", "value_parents" => ("Seg_sA")),
    Dict("child" => "Seg_V", "value_parents" => ("Seg_sV")),

]


shared_parameters = Dict(
    "sAV_initial_mean" => (0, [("Seg_sA", "initial_mean"),("Seg_sV", "initial_mean"), ("FF_sAV", "initial_mean")]),
    "sAV_drift" => (0, [("Seg_sA", "drift"),("Seg_sV", "drift"), ("FF_sAV", "drift")]),
    "A_input_noise" => (0, [("FF_A", "input_noise"), ("Seg_A", "input_noise")]),
    "V_input_noise" => (0, [("FF_V", "input_noise"), ("Seg_V", "input_noise")]),
    "sAV_initial_precision" => (0, [("Seg_sA", "initial_precision"),("Seg_sV", "initial_precision"), ("FF_sAV", "initial_precision")]),
)

# fix init_precision to 100


#Initialize the HGF
hgf = init_hgf(
    input_nodes = input_nodes,
    state_nodes = state_nodes,
    edges = edges,
    shared_parameters = shared_parameters,
)

# load in "merging_hgf_action.jl"
include("$(pwd())/action_functions/merging_hgf_action")

print(get_states(hgf))

agent_parameters = Dict(
    "action_noise" => 1,
    "p_common" => 0.5
)

agent = init_agent(
    merging_hgf,
    parameters = agent_parameters,
    substruct = hgf,
)

get_parameters(agent)

priors = Dict(
    ("FF_A", "input_noise") => Normal(-2, 1),
    ("FFV", "input_noise") => Normal(-2, 1),
    ("Seg_A", "input_noise") => Normal(-2, 1),
    ("Seg_V", "input_noise") => Normal(-2, 1),
    "p_common" => Beta(1,1),
    "action_noise" => LogNormal(0,1),
)


# SIMULATION
# Inputs

values = [-22, -11, 0, 11, 22]
## Number of samples
num_samples = 1000
## Generate samples
actions = Array(rand(values, num_samples))
## To view the first few samples
println(actions[1:10])
# Generate samples of vectors consisting of two values
inputs = [[v..., v...] for v in [rand(values, 2) for _ in 1:num_samples]]
# To view the first few samples
println(inputs[1:10])

reset!(agent)
give_inputs!(agent, inputs)

plot_trajectory(agent, "FF_A")
plot_trajectory!(agent, "FF_V")
plot_trajectory!(agent, "Seg_A")
plot_trajectory!(agent, "Seg_V")
plot_trajectory!(agent, "Ind_sV")
plot_trajectory!(agent, "C")
plot_trajectory!(agent, "action")

action_history = get_history(agent, "action")

action_history

# fit simulated data

# TRYING WITH Dataset

# load in dataset
dataset = CSV.read("park_and_kayser2023.csv", DataFrame)

show(dataset)

#doin some cleaning to get the numerical values
dataset[!, "input"] = JSON.parse.(dataset[!, "input"])

typeof(dataset[!,"input"])

typeof(dataset[!,"action"])

show(dataset)

inputs = dataset[:,1]

inputs

reset!(agent)
give_inputs!(agent, inputs)

plot_trajectory(agent, "xA")
plot_trajectory!(agent, "xV")
plot_trajectory!(agent, "FF_sAV")
plot_trajectory!(agent, "Ind_sA")
plot_trajectory!(agent, "Ind_sV")
plot_trajectory!(agent, "C")
plot_trajectory!(agent, "action")

action_history = get_history(agent, "action")

action_history

# Fitting model with all experiments multilevel

# I try with experiment 1 only
df_exp1 = dataset[dataset[!, "experiment"] .== "experiment 1", :]
inputs_exp1 = df_exp1[:,1]
actions_exp1 = df_exp1[:,2]

actions_exp1

reset!(agent)
give_inputs!(agent, inputs_exp1)

plot_trajectory(agent, "xA")
plot_trajectory!(agent, "xV")
plot_trajectory!(agent, "action")

action_history = get_history(agent, "action")
action_history_exp1 = action_history[2:length(action_history)]
action_history_exp1 = ForwardDiff.value.(action_history_exp1)


action_history_exp1
print(action_history_exp1)

# save csv of inputs_exp1 and actions_history_exp1
df_exp1_simulated_actions = DataFrame(
    input = inputs_exp1,
    action = action_history_exp1,
)

df_exp1_simulated_actions

CSV.write("df_exp1_simulated_actions.csv", df_exp1_simulated_actions)

chains = fit_model(
    agent,
    priors,
    inputs_exp1,
    actions_exp1,
    n_iterations = 2000,
)

