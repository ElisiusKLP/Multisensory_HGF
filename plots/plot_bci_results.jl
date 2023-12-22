"""
Creating a plot of action distribution for each audiovisual location condition

"""

using DataFrames
using Plots
using CSV
using JSON
using ActionModels, HierarchicalGaussianFiltering

# 
results = deserialize(open("/Users/elisius/github/bachelor/bci_fit_21-12-23.jls", "r"))

# load data
data = CSV.read("/Users/elisius/github/bachelor/park_and_kayser2023.csv", DataFrame)

# subset experiment 1 
data = data[data[!, "experiment"] .== "experiment 1", :]

enumerate(results)

x = results[String15("experiment 1"), String31("participant 1.13")]

plot(x)
get_posteriors(x, type="median")


# collect all participant chain posterior and find mean
#init arrays


for chain in results
    println("chain",chain)

    data = chain[2]

    post = get_posteriors(data, type = "median")
    println(post)
end

auditory_locations = [item[1] for item in inputs_exp1]
visual_locations = [item[2] for item in inputs_exp1]

df = DataFrame(auditory_locations = auditory_locations, visual_locations = visual_locations, action = action_history_exp1)

# Plotting
plot(
    df,
    group = :Action,
    x = :Auditory,
    y = :Visual,
    seriestype = :scatter,
    markersize = 5,
    markerstrokewidth = 0,
    legend = :topleft,
    xlabel = "Auditory Location",
    ylabel = "Visual Location",
    title = "Distribution of Actions",
)