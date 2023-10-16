
#byg hgf som i script 
# foder med random data
# prøv at lave parameter recovery
# prøv at læs dataset ind og fit det og se hvad der sker
# multilevel el. 3. HGF der kigger på den 

#List of input nodes to create
#List of input nodes to create
input_nodes = [Dict(
    “name” => “A”,
), Dict(“name” => “V”,)]
#List of state nodes to create
state_nodes = [
    Dict(
        “name” => “location”,
    ),
]
#List of child-parent relations
edges = [
    Dict(
        “child” => “A”,
        “value_parents” => (“location”),
    ),
    Dict(
        “child” => “V”,
        “value_parents” => (“location”),
    ),
]
#Initialize the HGF
hgf = init_hgf(
    input_nodes = input_nodes,
    state_nodes = state_nodes,
    edges = edges,
)
function multisensory_hgf_action(agent::Agent, input)
    action_noise = agent.parameters[“action_noise”]
    #Update hgf
    hgf = agent.substruct
    update_hgf!(hgf, input)
    #get out inferred location
    inferred_location = get_states(hgf, (“location”, “posterior_mean”))
    #Create action distribution
    action_distribution = Normal(inferred_location, action_noise)
    return action_distribution
end
agent_parameters = Dict(
    “action_noise” => 1
)
agent = init_agent(
    multisensory_hgf_action,
    parameters = agent_parameters,
    substruct = hgf,
)
reset!(agent)
get_parameters(agent)
give_inputs!(agent, [[0,5], [1,1], [0,3]])
plot_trajectory(agent, “location”)
plot_trajectory!(agent, “action”)
inputs = [[0,5], [1,1], [0,3]]
actions = [2, 1, 1.5]
priors = Dict(
    (“V”, “evolution_rate”) => Normal(-2, 1),
    (“A”, “evolution_rate”) => Normal(-2, 1),
)
results = fit_model(
    agent,
    priors,
    inputs,
    actions
)
plot(results)