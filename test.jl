
2+2

using DataFrames, CSV, 
using Distributions, 
using Plots, StatsPlots, 
using ActionModels, HierarchicalGaussianFiltering

agent = premade_agent("binary_rw_softmax")

get_parameters(agent)

parameters = Dict(
    "learning_rate" => 0.5
)

set_parameters!(agent, parameters)

get_parameters(agent) # the new one should be outputtet into console

input = 0 
#or an array
input = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]

reset!(agent)

give_inputs!(agent, input)

plot_trajectory(agent, "value_probability") # look in the plot and see how the predictions of the next value is modified
plot!(input, linetype = :scatter)

## instead of doing simulation we try to find the parameters such that they arent specified but predicted by the ActionModels

agent = premade_agent("binary_rw_softmax")


inputs = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
actions = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0]

param_priors = Dict("learning_rate" => Uniform(0, 1),
                    "action_precision" => Uniform()
)


results =
    fit_model(
        agent, # needs agent, param_priors, inputs, actions, in this order
        param_priors, 
        inputs, 
        actions, 
        n_chains = 3, n_iterations = 1000, verbose = false)



plot(chains)
plot(results)

plot_parameter_distribution(chains, param_priors)
plot_parameter_distribution(results, param_priors)

get_posteriors(results)
get_posteriors(results)

posterior_parameters = get_posterior(results)
set_parameters(agent, posterior_parameters) # this is bascially simulating with the posterior paramteres we pulled out just above

# fitting data to a model 
agent = premade_agent("binary_rw_softmax")
​
data = vcat(
    DataFrame(
        input = [1, 0, 1],
        action = [1, 0, 1],
        ID = 1,
        group = "A",
        experiment = "1",
    ),
    DataFrame(
        input = [1, 0, 1],
        action = [1, 0, 1],
        ID = 2,
        group = "A",
        experiment = "1",
    ),
    DataFrame(
        input = [1, 0, 1],
        action = [1, 0, 1],
        ID = 3,
        group = "B",
        experiment = "1",
    ),
    DataFrame(
        input = [1, 0, 1],
        action = [1, 0, 1],
        ID = 4,
        group = "B",
        experiment = "1",
    ),
    DataFrame(
        input = [1, 0, 1],
        action = [1, 0, 1],
        ID = 3,
        group = "C",
        experiment = "1",
    ),
    DataFrame(
        input = [1, 0, 1],
        action = [1, 0, 1],
        ID = 4,
        group = "C",
        experiment = "1",
    ),
    DataFrame(
        input = [1, 0, 1],
        action = [1, 0, 1],
        ID = 1,
        group = "A",
        experiment = "2",
    ),
    DataFrame(
        input = [1, 0, 1],
        action = [1, 0, 1],
        ID = 2,
        group = "A",
        experiment = "2",
    ),
    DataFrame(
        input = [1, 0, 1],
        action = [1, 0, 1],
        ID = 3,
        group = "B",
        experiment = "2",
    ),
    DataFrame(
        input = [1, 0, 1],
        action = [1, 0, 1],
        ID = 4,
        group = "B",
        experiment = "2",
    ),
)
​
    independent_group_cols = [:experiment]
    multilevel_group_cols = [:ID, :group]
    input_cols = [:input]
    action_cols = [:action]
​
    priors = Dict(
        "learning_rate" => Multilevel(
            :ID,
            LogitNormal,
            ["learning_rate_ID_mean", "learning_rate_ID_sd"],
        ),
        "learning_rate_ID_mean" => Multilevel(
            :group,
            Normal,
            ["learning_rate_group_mean", "learning_rate_group_sd"],
        ),
        "learning_rate_ID_sd" => LogNormal(0, 1),
        "learning_rate_group_sd" => LogNormal(0, 1),
        "learning_rate_group_mean" => Normal(0, 1),
    )
​
    results = fit_model(
        agent,
        priors,
        data;
        independent_group_cols = independent_group_cols,
        multilevel_group_cols = multilevel_group_cols,
        input_cols = input_cols,
        action_cols = action_cols,
        n_cores = 4,
        n_iterations = 1000,
        n_chains = 2,
    )


results["1"]

# Theres also tutorials inside the HGF github
## classic_usdchf
## choose prior through prior predictive simulations 
## nice at starte med parameters recovery inden du fatter til ægte data
## vi har simuleret nogle judgements/acitons ud fra locations der var i virkeligheden
## vi kan så prøve at predicte de parametre der er brugt til at lave de judgements
## og bruge de parameters til at recover de priors vi brugte til
## du kan løse det hul der er i hgfen ved at stramme priors
## brug noget tid på at simulere
## installer Turing - Using Turing

## use a HGF to model which of two HGF is the most probable

# make a HGF

input_nodes = [
    Dict("name" => "A"),
    Dict("name" => "V")
    ]

state_nodes = [
    Dict(
        "name" => "location"
    )
]

edges = [
    Dict(
        "child" => "A",
        "value_parent" => "location"
    ),
    Dict(
        "child" => "V",
        "value_parent" => "location"
    )
] 

hgf = init_hgf(input_nodes = input_nodes,
state_nodes = state_nodes,
edges = edges)

update_hgf!(hgf, [1, 0.5])

plot_trajectory(hgf, ("location, "posterior"))

# we need to make an aciton function which turns the HGF into an agent

function multisensory_hgf_actionmodel(agent, input)

    hgf = agent.substruct
    agetn
    update 
end

