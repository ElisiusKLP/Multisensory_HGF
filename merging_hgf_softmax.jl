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
forced_fusion
# FF_sAV
# FF_xA
# FF_xV

Independet
# Ind_sA
# Ind_sV
# Ind_xA
# Ind_xV

model_combination
# C - cause



edges, nodes

init_hgf

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


#Initialize the HGF
hgf = init_hgf(
    input_nodes = input_nodes,
    state_nodes = state_nodes,
    edges = edges,
)

print( get_parameters(hgf) )

print( get_states(hgf) )

get_surprise(hgf)

function merging_hgf(
    agent::Agent, 
    input, constant_cue="A", 
    internal_variables = "clean", 
    decision="model_averaging_surprise"
    )

    # generate latent noise for the internal variables
    auditory_latent_noise = randn(1)[1]
    visual_latent_noise = randn(1)[1]

    if internal_variables == "noisy"
        if length(input) == 5
            FF_auditory_stimulus = input[1] * auditory_latent_noise
            FF_visual_stimulus = input[2] * visual_latent_noise

            Seg_auditory_stimulus = input[3] * auditory_latent_noise
            Seg_visual_stimulus = input[4] * visual_latent_noise

            cue = input[5]
        else
            FF_auditory_stimulus = input[1] * auditory_latent_noise
            FF_visual_stimulus = input[2] * visual_latent_noise

            Seg_auditory_stimulus = input[3] * auditory_latent_noise
            Seg_visual_stimulus = input[4] * visual_latent_noise

            cue = constant_cue
        end
    elseif internal_variables == "clean"
        if length(input) == 5
            FF_auditory_stimulus = input[1]
            FF_visual_stimulus = input[2]

            Seg_auditory_stimulus = input[3]
            Seg_visual_stimulus = input[4]

            cue = input[5]
        else
            FF_auditory_stimulus = input[1]
            FF_visual_stimulus = input[2]

            Seg_auditory_stimulus = input[3]
            Seg_visual_stimulus = input[4]

            cue = constant_cue
        end
    end

    action_noise = agent.parameters["action_noise"]
    prior_common = agent.parameters["p_common"]

    hgf = agent.substruct

    # update HGF with inputs
    update_hgf!(hgf, [FF_auditory_stimulus, FF_visual_stimulus, Seg_auditory_stimulus, Seg_visual_stimulus])

    # Forced Fusion (Common Cause)
    FF_inferred_location = get_states(hgf, ("FF_sAV", "posterior_mean"))

    ###FF_action_distribution = Normal(FF_inferred_location, action_noise)

    # Segregation (Independent Causes)
    if cue == "A"
        Ind_inferred_location = get_states(hgf, ("Seg_sA", "posterior_mean"))
    elseif cue == "V"
        Ind_inferred_location = get_states(hgf, ("Seg_sV", "posterior_mean"))
    end

    ###Ind_action_distribution = Normal(Ind_inferred_location, action_noise)

    # CAUSE estimation
    ## calculating likelihoods

    #FF_likelihood_auditory = pdf(MvNormal(FF_inferred_location, FF_inferred_location]))
 #= 
    Ind_likelihood_auditory = pdf(Ind_action_distribution, auditory_stimulus)
    Ind_likelihood_visual = pdf(Ind_action_distribution, visual_stimulus)
    likelihood_independent = Ind_likelihood_auditory * Ind_likelihood_visual
 =#

    ## Model Averaging using surprise
    surpriseA_common_cause = get_surprise(hgf, ("FF_A"))
    surpriseV_common_cause = get_surprise(hgf, ("FF_V"))
    surprise_common_cause = surpriseA_common_cause + surpriseV_common_cause

    surpriseA_independent = get_surprise(hgf, ("Seg_A"))
    surpriseV_independent = get_surprise(hgf, ("Seg_V"))
    surprise_independent = surpriseA_independent + surpriseV_independent

    soft_array = [-surprise_common_cause, -surprise_independent]
    softmax = softmax!(soft_array)

    # I call this posterior even though i haven't multiplied by the prior
    # I asumme the model accumulates evidence into the suprise information
    posterior_common = softmax[1]
    posterior_independent = softmax[2]

    cause_probability_ratio = posterior_common / posterior_independent

    # DECISION
    ## Model averaging w. surprise
    if decision == "model_averaging_surprise"
        sV_hat = posterior_common * FF_inferred_location + posterior_independent * Ind_inferred_location
        sA_hat = posterior_common * FF_inferred_location + posterior_independent * Ind_inferred_location
    end
    ## 

    # Generating action
    if cue == "A"
        action = Normal(sA_hat, action_noise)
    elseif cue == "V"
        action = Normal(sV_hat, action_noise)
    end

    return action
end




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

priors = Dict(
    ("xV", "input_noise") => Normal(-2, 1),
    ("xA", "input_noise") => Normal(-2, 1),
    "action_noise" => LogNormal(0,0.2),
)

# simulate some data
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

# simulate some data

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

