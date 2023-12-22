"""
Creating a plot of action distribution for each audiovisual location condition

"""

using DataFrames
using Plots
using CSV
using JSON

# load data csv
data = CSV.read("/Users/elisius/github/bachelor/df_exp1_simulated_actions.csv", DataFrame)

data[ !, "action"][1]
typeof(data[!, "input"])

# Assuming 'data' is your DataFrame and 'column_name' is the name of the column you want to clean
data[!, "input"] = replace.(data[!, "input"], "Any" => "")
data[!, "input"] = replace.(data[!, "input"], "Any" => "")


data[!, "input"] 
data[!, "action"]

data[!, "input"] = JSON.parse.(data[!, "input"])
data[!, "action"] = JSON.parse.(data[!, "action"])

input

inputs_exp1 = data[!, "input"] = JSON.parse.(data[!, "input"])

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