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


# Simulate some data with four columns one locV, locA, respV, respA
# 1000 rows
# locV and locA are the locations of the visual and auditory stimuli
# respV and respA are the responses to the visual and auditory stimuli
# there should be locations: -22, -11, 0, 11, 22 for both locV and locA
# there should be responses: 

function simulate(n_trials, p_common, sigA, sigV, sigP)
    
    # locV and locA are the locations of the visual and auditory stimuli
    # respV and respA are the responses to the visual and auditory stimuli
    data = Dict(
        :locV,
        :locA,
        :respV,
        :respA,
    )
    

    # P(C=1) is the prob of a common cause
    for  in 1:1000:
        c = Binomial(1, p_common)
        C = rand(c)
    
        if C == 0
            S_AV = Normal(0, sigP)
            push!(data[:respA], S_AV)
            push!(data[:respV], S_AV)

        if C == 1
            S_A = Normal(0, sigP)
            S_V = Normal(0, sigP)
            push!(data[:respA], S_A)
            push!(data[:respV], S_V)

        X_V = Normal(S_V, sigV)
        X_A = Normal(S_A, sigA)
        
        push!(data[:locV], X_V)
        push!(data[:locA], X_A)
        
    end

end


# simulate data
simulate(1000, 0.5, 1, 1, 1)

# run the BCI_fit_model script
include("BCI_fit_model.jl")

# run the fit_model function from BCI_fit_model script
fit_model()








#Common, independent
# prior for common: P_common

#Hvis common: location S_AV
# Gaussian(mean_AV, sigP) 


#PArametrs: 
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




original_action_model = function(agent, inputs)
    auditoryinput[1]
    visual_stimulus
    cue
 

    #Udregn S_A og S_V
    # både for common og ikke common
    S_AV_if_common = EQ4 # everything is either observations or parameters

    S_A_if_common = S_AV_if_common
    S_V_if_common = S_AV_if_common
    
    S_A_if_independent = EQ5
    S_V_if_independent = EQ6
    
    
    # Udregn p_c
    prob_obsA_if_common = pdf(obsA, normal(S_A_if_common, sigA))
    prob_obsV_if_common = pdf(obsV, normal(S_V_if_common, sigV))
    prob_obsA_obsV_if_common = prob_obsA_if_common * prob_obsV_if_common
    
    posterior_common = prob_obsA_obsV_if_common * prior_common / prob_obsA_obsV_general
    
    
    # Weigh S_A og S_V med p_c (model averaging etc)
    #Model averaging
    S_A_averaged = p_common * S_A_if_common + (1 - p_common) * S_A_if_independent
    S_V_averaged = p_common * S_V_if_common + (1 - p_common) * S_V_if_independent
    
    #Model selection
    S_A_selected = (p_common > 0.5) * S_A_if_common + (p_common <= 0.5) * S_A_if_independent
    S_V_selected = (p_common > 0.5) * S_V_if_common + (p_common <= 0.5) * S_V_if_independent

    
end

original_params = Dict(
    #PArametrs: 
    # p_common - prior for Common or not
    # muP - centrality bias
    # sigP - position variability from 0
    # sigA - auditory noise
    # sigV - visual noise
)

original_states = Dict(
    #States:
# C - whether common or not
# S_AV - the shared position
# S_A - the auditory position
# S_V - the visual position
)

agent = init_agent(
    original_action_model,
    parameters = original_params,
    states = original_states
)


get_parameters(agent)












----




forced_fusionc
# FF_S_V
#FF_S_A

Independet
 #IND_S_V


model_comp




edges, nodes



init_hgf