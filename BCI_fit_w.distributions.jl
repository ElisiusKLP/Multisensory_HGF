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
        auditory_stimulus = input[1]
        visual_stimulus = input[2]
        cue = input[3]
    else
        auditory_stimulus = input[1]
        visual_stimulus = input[2]
        cue = constant_cue
    end

    #Get parameters
    p_common = agent.parameters["p_common"]
    muP = agent.parameters["muP"]
    sigP = agent.parameters["sigP"]
    sigA = agent.parameters["sigA"]
    sigV = agent.parameters["sigV"]

    # variances of A and V and prior
    varP = sigP^2
    varA = sigA^2
    varV = sigV^2

    # variances of estimates given common or independent
    #= varVA_hat = 1 / ( 1 / varV + 1 / varA + 1 / varP )
    varV_hat = 1 / ( 1 / varV + 1 / varP )
    varA_hat = 1 / ( 1 / varA + 1 / varP ) =#
    # I drop this as it is just confusing code practice when realting it to the used equations in papers

    # variances used in computing probasbility of common or independent causes
    #=     var_common = varV * varA + varV * varP + varA * varP
    varV_independent = varV + varP
    varA_independent = varA + varP =#

    # Calculate estimates sAV and sA and sV (forces fusion and segreated)
    # både for common og ikke common
    sAV_hat_if_common = ( (auditory_stimulus / varA) + (visual_stimulus / varV) + ( muP / varP ) ) / ( ( 1 / varA) + ( 1 / varV) + ( 1 / varP) ) # everything is either observations or parameters

    sA_hat_if_common = sAV_hat_if_common
    sV_hat_if_common = sAV_hat_if_common
    
    S_A_if_independent = ( (auditory_stimulus / varA) + ( muP / varP ) ) / ( ( 1 / varA) + ( 1 / varP) ) # everything is either observations or parameters
    S_V_if_independent = ( (visual_stimulus / varV) + ( muP / varP ) ) / ( ( 1 / varV) + ( 1 / varP) ) # everything is either observations or parameters
  
    
    # create probability distributions for the estimates with mean 0 to use in likelihood
    likelihood_common = pdf( MvNormal(zeros(3), [varA, varV, varP]), [S_A_if_independent, S_V_if_independent, muP] )
    likelihood_independent =pdf( MvNormal(zeros(2), [varA, varV]), [S_A_if_independent, S_V_if_independent] )

    
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

    return posterior_C, sV_hat, sA_hat
end


original_params = Dict(
    #PArametrs: 
    "p_common" => 1 ,# p_common - prior for Common or not
    "muP" => 0,# muP - centrality bias
    "sigP" => 1,# sigP - position variability from 0
    "sigA" => 1,# sigA - auditory noise
    "sigV" => 1,# sigV - visual noise
)

#States:
# C - whether common or not
# S_AV - the shared position
# S_A - the auditory position
# S_V - the visual position
original_states = Dict(
    "name" => "C",
    "name" => "S_AV",
    "name" => "S_A",
    "name" => "S_V"
)

priors = Dict(
    "p_common" => Normal(0,1),
    "sigP" => Normal(0, 1),
    "sigA" => Normal(0, 1),
    "sigV" => Normal(0, 0.5),
    "muP" => Normal(0, 1),
)

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

# Fitting the model to the simulated data
fit_model(
    agent,
    priors,
    inputs,
    actions,
    n_iterations = 2000
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