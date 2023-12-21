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


# I dont know if i need to convert the sigmas to variances

original_action_model = function(agent::Agent, input, constant_cue = "A", decision = "model_averaging")
    """ 
    constant_cue can be either "A" or "V" and is used to specify whether the cue is auditory or visual
    decision can be either "model_averaging" or "model_selection" and is used to specify whether the decision rule is model averaging or model selection
    """

    if length(input) == 3
        auditory_stimulus = input[1] * randn(1) # added random noise
        visual_stimulus = input[2] * randn(1) # added random noise
        cue = input[3]
    else
        auditory_stimulus = input[1] * randn(1) # added random noise
        visual_stimulus = input[2] * randn(1) # added random noise 
        cue = constant_cue
    end

    #Get parameters
    p_common = agent.parameters["p_common"]
    muP = agent.parameters["muP"]
    sigP = agent.parameters["sigP"]
    sigA = agent.parameters["sigA"]
    sigV = agent.parameters["sigV"]
    action_noise = agent.parameters["action_noise"]

    # variances of A and V and prior
    varP = sigP^2
    varA = sigA^2
    varV = sigV^2

    # variances of estimates given common or independent
    #= varVA_hat = 1 / ( 1 / varV + 1 / varA + 1 / varP )
    varV_hat = 1 / ( 1 / varV + 1 / varP )
    varA_hat = 1 / ( 1 / varA + 1 / varP ) =#
    # I drop this as it is just confusing code practice when realting it to the used equations in papers
    # It is instead part of the sAV_hat_if_common equation

    # variances used in computing probasbility of common or independent causes
    var_common = varV * varA + varV * varP + varA * varP
    varV_independent = varV + varP
    varA_independent = varA + varP

    # Calculate estimates sAV and sA and sV (forces fusion and segreated)
    # både for common og ikke common
    sAV_hat_if_common = ( (auditory_stimulus / varA) + (visual_stimulus / varV) + ( muP / varP ) ) / ( ( 1 / varA) + ( 1 / varV) + ( 1 / varP) ) # everything is either observations or parameters

    sA_hat_if_common = sAV_hat_if_common
    sV_hat_if_common = sAV_hat_if_common
    
    S_A_if_independent = ( (auditory_stimulus / varA) + ( muP / varP ) ) / ( ( 1 / varA) + ( 1 / varP) ) # everything is either observations or parameters
    S_V_if_independent = ( (visual_stimulus / varV) + ( muP / varP ) ) / ( ( 1 / varV) + ( 1 / varP) ) # everything is either observations or parameters
    
    # udregn prob of common or independent
    ## this is a weighted distance metric
    quad_common = ( visual_stimulus - auditory_stimulus )^2 * varP + ( visual_stimulus - muP )^2 * varA + ( auditory_stimulus - muP )^2 * varV
    quadV_independent = ( visual_stimulus - muP )^2
    quadA_independent = ( auditory_stimulus - muP )^2

    # likelihood of observations (xV, xA) given C (1=common or 2=independent)
    ## this is the PDF
    likelihood_common = exp(-quad_common/(2*var_common)) / (2*pi*sqrt(var_common))
    likelihoodV_independent = exp(-quadV_independent/(2*varV_independent)) / sqrt(2*pi*varV_independent)
    likelihoodA_independent = exp(-quadA_independent/(2*varA_independent)) / sqrt(2*pi*varA_independent)
    likelihood_independent = likelihoodV_independent * likelihoodA_independent

    # posterior probability of state C (cause 1 or 2) given observations (xV, xA)
    posterior_common = likelihood_common * p_common
    posterior_independent = likelihood_independent * (1 - p_common)
    posterior_C = posterior_common / ( posterior_common + posterior_independent )
    
    # DECISION RULE
    if decision == "model_averaging"
        sV_hat = posterior_C * sV_hat_if_common + (1 - posterior_C) * S_V_if_independent
        sA_hat = posterior_C * sA_hat_if_common + (1 - posterior_C) * S_A_if_independent
    elseif decision == "model_selection"
        sV_hat = (posterior_C > 0.5) * sV_hat_if_common + (posterior_C <= 0.5) * S_V_if_independent
        sA_hat = (posterior_C > 0.5) * sA_hat_if_common + (posterior_C <= 0.5) * S_A_if_independent
    end

    if cue == "A"
        action = Normal(sA_hat, action_noise)
    elseif cue == "V"
        action = Normal(sV_hat, action_noise)
    end

    return action
end

# If i wanted to save the posterior_C 
#agent.states["C"] = posterior_C
#push!(agent.history["C"], posterior_C)

original_params = Dict(
    #PArametrs: 
    "p_common" => 1 ,# p_common - prior for Common or not
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
    "sigP" => Normal(0, 1),
    "sigA" => Normal(0, 1),
    "sigV" => Normal(0, 1),
    "muP" => Normal(0, 1),
    "action_noise" => LogNormal(0,0.2),
)
# prøv at fit uden muP
# og smallere priors


agent = init_agent(
    original_action_model,
    parameters = original_params,
    states = original_states
)

get_parameters(agent)

# simulate some data
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

# REAL DATA

dataset = CSV.read("park_and_kayser2023.csv", DataFrame)

show(dataset)

dataset[!, "input"] = JSON.parse.(dataset[!, "input"])

typeof(dataset[!,"input"])

typeof(dataset[!,"action"])

inputs = dataset[:,1]

# I try with experiment 1 only
df_exp1 = dataset[dataset[!, "experiment"] .== "experiment 1", :]
inputs_exp1 = df_exp1[:,1]
actions_exp1 = df_exp1[:,2]

reset!(agent)
give_inputs!(agent, inputs_exp1)

plot_trajectory(agent, "action")

agent.history

# Fitting the model to the simulated data
chains = fit_model(
    agent,
    priors,
    inputs_exp1,
    actions_exp1,
    n_iterations = 2000,
)

serialize("bci_fit_19-12-23.jls", chains)

plot(chains)



auditory_input, visual_input

input_cols = [:auditory_input, :visual_input]
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