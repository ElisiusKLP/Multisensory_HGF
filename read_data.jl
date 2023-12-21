
using MAT
using DataFrames

path = "/Users/elisius/github/bachelor/data/DataExp1to10.mat"

data = matread(path)

@show keys(data)
@show keys(data["Dataset"])
@show keys(data["Dataset"][1]) # 1st experiment
@show keys(data["Dataset"][1][1]) # 1st subject

dataset = data["Dataset"]

dataset[1][1][:, 1]

# Initialize arrays to store the data
input_array = []
action_array = []
subject_array = String[]
experiment_array = String[]

enumerated = enumerate(dataset)
enumerated]
# Iterate over experiments

for (experiment_index, experiment) in enumerate(dataset)
    
    for (participant_index, participant) in enumerate(experiment)
        for i in 1:size(participant, 1)
            A = participant[i, 2]
            V = participant[i, 1]
            AV = [A, V]
            push!(input_array, AV)
            push!(action_array, participant[i, 5])
            push!(subject_array, "participant $participant_index")
            push!(experiment_array, "experiment $experiment_index")
        end
    end
end

input_array


# Create DataFrame
df = DataFrame(
    input = input_array,
    action = action_array,
    subject = subject_array,
    experiment = experiment_array,
)

# Display the DataFrame
show(df)
df[!,"input"]

typeof(df[!,"input"])

# Save DataFrame to csv
CSV.write("park_and_kayser2023.csv", df)

#------------------------
"""
Creating a new dataset with the auditory and visual input as separate columns
"""

# Initialize arrays to store the data
auditory_location = []
visual_location = []
action_array = []
subject_array = String[]
experiment_array = String[]

enumerated = enumerate(dataset)
enumerated]
# Iterate over experiments

for (experiment_index, experiment) in enumerate(dataset)
    
    for (participant_index, participant) in enumerate(experiment)
        for i in 1:size(participant, 1)
            A = participant[i, 2]
            V = participant[i, 1]
            
            push!(auditory_location, A)
            push!(visual_location, V)
            push!(action_array, participant[i, 5])
            push!(subject_array, "participant $experiment_index.$participant_index")
            push!(experiment_array, "experiment $experiment_index")
        end
    end
end




# Create DataFrame
df = DataFrame(
    auditory_location = auditory_location,
    visual_location = visual_location,
    action = action_array,
    subject = subject_array,
    experiment = experiment_array,
)

# Display the DataFrame
show(df)
df[!,"input"]

typeof(df[!,"input"])

# Save DataFrame to csv
CSV.write("park_and_kayser2023.csv", df)

