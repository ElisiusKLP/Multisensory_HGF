"""
Creating Segregated model with HGF.
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
input_nodes = [
    Dict(
    "name" => "X_A",
), Dict("name" => "X_V",)]

#List of state nodes to create
state_nodes = [
    Dict(
        "name" => "S_A",
    ), Dict(
        "name" => "S_V",
    )
]

#List of child-parent relations
edges = [
    Dict(
        "child" => "X_A",
        "value_parents" => ("S_A"),
    ),
    Dict(
        "child" => "X_V",
        "value_parents" => ("S_V"),
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
    auditory_stimulus = input[1]
    visual_stimulus = input[2]
    cue = input[3]
    
    action_noise = agent.parameters["action_noise"]
    #Update hgf
    hgf = agent.substruct

    update_hgf!(hgf, [auditory_stimulus, visual_stimulus])

    if cue == "auditory"

        inferred_position = get_states(hgf, ("S_A", "posterior_mean"))

    else if cue == "visual"

        inferred_ppsition = get_states(hgf, ("S_V", "posterior_mean"))

    end
    
    action_distribution = Normal(inferred_position, action_noise)

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
    ("X_V", "input_noise") => Normal(-2, 1),
    ("X_A", "input_noise") => Normal(-2, 1),
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
print(get_parameters(agent))
set_parameters!(
    agent,
    Dict(
        ("V", "evolution_rate") => -5,
        ("A", "evolution_rate") => 2,
    )
)
get_parameters(agent)
give_inputs!(agent, inputs)

plot_trajectory(agent, "X_A")
plot_trajectory!(agent, "X_V")
plot_trajectory!(agent, "S_A")
plot_trajectory!(agent, "S_V")
plot_trajectory!(agent, series) # Somethings wrong here

action_history = get_history(agent, "action")

action_history = action_history[2:1001]

init_params = get_parameters(agent)

# I can create an array or times series object to convert in to the right format but maybe this has to be done in
series = [([i for i in 1:length(action_history)], [x[1] for x in action_history], [x[2] for x in action_history])]

series

get_history(agent, "")
# FITTING MODEL from scratch


results = fit_model(
    agent,
    priors,
    inputs,
    action_history
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