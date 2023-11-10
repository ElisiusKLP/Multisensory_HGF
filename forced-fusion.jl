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

function multisensory_hgf_action(agent::Agent, input)
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
actions = rand(values, num_samples)
## To view the first few samples
println(actions[1:10])
# Generate samples of vectors consisting of two values
inputs = [rand(values, 2) for _ in 1:num_samples]
# To view the first few samples
println(inputs[1:10])

# Feeding the agent with inputs to generate some actions

reset!(agent)
give_inputs!(agent, inputs)

plot_trajectory(agent, "A")
plot_trajectory!(agent, "V")
plot_trajectory!(agent, "location")
plot_trajectory!(agent, "action")

action_history = get_history(agent, "action")

action_history = action_history[2:1001]

init_params = get_parameters(agent)

get_history(agent, "")
# FITTING MODEL from scratch


results = fit_model(
    agent,
    priors,
    inputs,
    action_history,
    n_cores = 2,
)


plot(results)

x = get_posteriors(results, type = "median")
x["V", "evolution_rate"]

get_parameters(hgf)
get_parameters(agent)
ActionModels.get_parameters(agent)

CSV.write("", results) ## write csv

serialize("forces-fusion-26-10-23.jls", results) # saveing the mcmc chains object (model)

results = deserialize("forces-fusion-26-10-23.jls") # loading in serialized MCMCChain object

typeof(results)

plot(results, ("A","evolution_rate") )

# plot trajectory of agent locations and actions
reset!(agent)
give_inputs!(agent, inputs)
plot_trajectory(agent, "location")
plot_trajectory!(agent, "action")

histogram(inputs)
histogram(actions)

# the prior predictive simulation tells us the distribution of observed data we expect before we have observed any data

results

plot_parameter_distribution(results, priors)

plot_parameter_distribution(results)

plot_predictive_simulation(
    priors,
    agent,
    inputs,
    ("location", "posterior_mean");
    n_simulations = 3
)

plot_trajectory!(agent, "A")

plot_predictive_simulation(
    results,
    agent,
    inputs,
    ("action");
    n_simulations = 3
)

plot_trajectory!(agent, "action")

give_inputs!(agent, inputs)
reset!(agent)

plot_trajectory(agent, "action")

# Parameter recovery

## create a list of two lists for each parameter which we want to test
A_er = [-5,-2,0,2]
V_er = [-5,-2,0,2]

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

    key = string("V_",v_rate, "_A_", a_rate)

    z = 1

    while z <= 200 # number of iterations (could change to n if function)
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
            n_chains = 4,
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

key = string("V_",v_rate, "_A_", a_rate)
post_median = get_posteriors(results, type = "median")
if key in keys(posterior_medians)
    push!(posterior_medians[key], post_median)
else
    posterior_medians[key] = post_median
end

posterior_medians
# Creating a modified plot_predictive_simulation function which has a boolean argument to output the synthetic data

get_parameters(hgf)
get_parameters(agent)

histogram(actions_pred)
histogram!(action_history)
histogram(samples["V", "evolution_rate"])
histogram(samples["A", "evolution_rate"])
medians["V", "evolution_rate"]
medians["A", "evolution_rate"]

histogram(actions_pred)
histogram!(action_history)

# creating a new fitted model with the new action data

results2 = fit_model(
    agent,
    priors,
    inputs,
    actions_pred
)

# Synthesizing input and action data from fitted parameters

results2

plot_parameter_distribution(results2, priors)


get_parameters(agent)

results
results[1,1,2]

# Number of data points you want in your synthetic dataset
# Assuming posterior_samples is a 1000x14x2 array
chain1_samples = results[:,:,1]
chain2_samples = results[:,:,2]
all_samples = vcat(chain1_samples, chain2_samples)

posterior_samples

# Initialize arrays to store synthetic data
synthetic_x = zeros(n_points)  # Initialize synthetic input data
synthetic_y = zeros(n_points)  # Initialize synthetic output data

# Initialize arrays to store synthetic data
n_samples = size(all_samples, 1)
synthetic_x = Vector{Vector{Float64}}(undef, n_samples)
synthetic_y = Vector{Vector{Float64}}(undef, n_samples)

# Generate synthetic data using posterior samples
for i in 1:n_samples
    # Sample parameter values from the posterior
    β_sample = all_samples[i, 1:7]  # Adjust the indices for your model
    σ_sample = all_samples[i, 8:14]  # Adjust the indices for your model

    # Generate synthetic x values (2-length vectors)
    synthetic_x[i] = rand(Normal(0, 1), 2)  # Adjust the distribution as per your model

    # Generate synthetic y values using the sampled parameters and synthetic x
    synthetic_y[i] = β_sample .* synthetic_x[i] + rand(Normal(0, σ_sample))
end

# Create predictions from an already existing dataset

synthetic_results = fit_model(
    agent,
    priors,
    inputs,
    action_history
)


predictions = Turing.predict(
    agent,
    results
)

# try to load in the results from the fitted model using serialization
loaded = deserialize("forces-fusion-26-10-23.jls") # loading in serialized MCMCChain object