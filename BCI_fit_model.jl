"""
A model based on the BCI multisensory integration model

based on the study by:


"""
module BCI

export fitModel

# Function to simulate responses from the BCI model for a given parameter set
function fitModel(parameters, parameterNames, dataVA, responseLoc, decisionFun)
    # Fix the initializing random state
    srand(55)

    nIntSamples = 10000  # minimum required for model fitting

    # Checking dataVA
    reqDataVars = ["locV", "locA", "respV", "respA"]
    if any(x -> !(x in names(dataVA)), reqDataVars)
        error("bci_fitmodel:missingDataVariable", "Missing variable from dataVA in bci_fitmodel.")
    end
    
    # Ensure that only data are included where both A and V signals are used
    dataVA = dataVA[.!isnan.(dataVA.locV) .& !isnan.(dataVA.locA), :]

    # Check responseLoc
    if iscolumn(responseLoc)
        responseLoc = reshape(responseLoc, 1, :)
    end

    # Check parameters
    if iscolumn(parameters)
        parameters = reshape(parameters, 1, :)
    end

    # To save computation time, check for what stimulus conditions were
    # presented, then simulate each one


    # Parameters: [p_common(i), xP(m), sigP(m), kernelWidth(n), sigA(p), sigV(k)]
    p_common = parameters[ismember(parameterNames, "p_common")]  # Set to 1 for full integration, 0 for full segregation
    sigP = parameters[ismember(parameterNames, "sigP")]
    sigA = parameters[ismember(parameterNames, "sigA")]
    sigV = parameters[ismember(parameterNames, "sigV")]

    # Setting default for muP if it is not specified
    xP = haskey(parameterNames, "muP") ? parameters[ismember(parameterNames, "muP")] : 0

    # discrete responses?? i.e. response loc vector e.g. [1 2 3 4 5]
    # or continuous responseLoc = 1
    kernelWidth = length(responseLoc) == 1 ? parameters[ismember(parameterNames, "kW")] : 1

    # Throw an error if there is a missing parameter
    if any([isempty(p_common), isempty(xP), isempty(sigP), isempty(sigA), isempty(sigV), isempty(kernelWidth)])
        error("bci_fitmodel:missingParameter", "Missing parameter in bci_fitmodel.")
    end

    # Variances of A and V and prior
    varV = sigV^2
    varA = sigA^2
    varP = sigP^2

    # Variances of estimates given common or independent causes
    varVA_hat = 1 / (1 / varV + 1 / varA + 1 / varP)
    varV_hat = 1 / (1 / varV + 1 / varP)
    varA_hat = 1 / (1 / varA + 1 / varP)

    # Variances used in computing probability of common or independent causes
    var_common = varV * varA + varV * varP + varA * varP
    varV_indep = varV + varP
    varA_indep = varA + varP

    # Initialize variable to collect log-likelihood values for each condition
    logLikeCond = fill(NaN, nCond)

    # Simulate responses for each condition
    for indCond in 1:nCond
        sV = conditions[indCond, 1]
        sA = conditions[indCond, 2]

        # Generation of fake data
        xV = sV .+ sigV * randn(nIntSamples)
        xA = sA .+ sigA * randn(nIntSamples)

        # Estimates given common or independent causes
        s_hat_common = (xV / varV + xA / varA + xP / varP) * varVA_hat
        sV_hat_indep = (xV / varV + xP / varP) * varV_hat
        sA_hat_indep = (xA / varA + xP / varP) * varA_hat

        # Probability of common or independent causes
        quad_common = (xV - xA).^2 * varP + (xV - xP).^2 * varA + (xA - xP).^2 * varV
        quadV_indep = (xV - xP).^2
        quadA_indep = (xA - xP).^2

        # Likelihood of observations (xV, xA) given C, for C=1 and C=2
        likelihood_common = exp.(-quad_common / (2 * var_common)) / (2 * π * sqrt(var_common))
        likelihoodV_indep = exp.(-quadV_indep / (2 * varV_indep)) / sqrt(2 * π * varV_indep)
        likelihoodA_indep = exp.(-quadA_indep / (2 * varA_indep)) / sqrt(2 * π * varA_indep)
        likelihood_indep = likelihoodV_indep .* likelihoodA_indep

        # Posterior probability of C given observations (xV, xA)
        post_common = likelihood_common * p_common
        post_indep = likelihood_indep * (1 - p_common)
        pC = post_common ./ (post_common + post_indep)

        # Generate spatial location responses and compute loglike
        if decisionFun == 1
            # Mean of posterior - Model averaging
            # Overall estimates: weighted averages
            sV_hat = pC .* s_hat_common + (1 - pC) .* sV_hat_indep
            sA_hat = pC .* s_hat_common + (1 - pC) .* sA_hat_indep
        elseif decisionFun == 2
            # Model selection instead of model averaging
            sV_hat = (pC .> 0.5) .* s_hat_common + (pC .<= 0.5) .* sV_hat_indep
            sA_hat = (pC .> 0.5) .* s_hat_common + (pC .<= 0.5) .* sA_hat_indep
        elseif decisionFun == 3
            # Probability matching
            thresh = rand(nIntSamples)
            sV_hat = (pC .> thresh) .* s_hat_common + (pC .<= thresh) .* sV_hat_indep
            sA_hat = (pC .> thresh) .* s_hat_common + (pC .<= thresh) .* sA_hat_indep
    

        # compute predicted responses for discrete and continuous case
        if length(responseLoc) > 1
            # discrete responses
            lengthRespLoc = length(responseLoc)

            # find the response location closest to sV_hat and
            # sA_hat, ie with minimum deviation
            _, tV = argmin(abs.(sV_hat .- responseLoc), dims=2)
            _, tA = argmin(abs.(sA_hat .- responseLoc), dims=2)
            sV_pred_resp = responseLoc[tV]
            sA_pred_resp = responseLoc[tA]
        else
            # continuous responses used, no discretization
            sV_pred_resp = sV_hat
            sA_pred_resp = sA_hat
        end

        # A, V responses given by participant for particular A,V location combination
        dataV = dataConditions[indCond][:, 1]'  # transpose and convert to vector
        dataA = dataConditions[indCond][:, 2]'  # transpose and convert to vector

        # Compute loglike for A and V responses
        if length(responseLoc) > 1
            # discrete case
            # calculate frequencies of predictions, at least 0.00001 (1 out of 100,000)
            freq_predV = max.(0.00001, hist(sV_pred_resp, responseLoc) / nIntSamples)
            freq_predA = max.(0.00001, hist(sA_pred_resp, responseLoc) / nIntSamples)

            # calculate absolute frequencies of actual responses
            freq_dataV = hist(dataV, responseLoc)
            freq_dataA = hist(dataA, responseLoc)

            # calculate log-likelihood
            logLikeA = sum(freq_dataA .* log.(freq_predA))
            logLikeV = sum(freq_dataV .* log.(freq_predV))
        else
            # continuous case
            # gaussian kernel distribution for each condition
            # for each condition average over gaussian likelihoods
            logLikeA = log(mean(normpdf.(repeat(dataA, nIntSamples, 1), repeat(sA_pred_resp, 1, length(dataA)), kernelWidth)))
            logLikeV = log(mean(normpdf.(repeat(dataV, nIntSamples, 1), repeat(sV_pred_resp, 1, length(dataV)), kernelWidth)))
        end

        # sum loglike across different task responses
        if all(isnan.(logLikeA)) && all(isnan.(logLikeV))
            # In this case, the sum would be 0, which would mean that the
            # likelihood would be 1, which is not, so we prevent that
            # from happening.
            logLikeCond[indCond] = NaN
        else
            logLikeCond[indCond] = nansum([nansum(logLikeA) nansum(logLikeV)])
        end

        # for a particular parameter setting make plots and save biases etc.
        if nargout > 1
            # store values
            mdlEval = Dict(
                :sA_resp => sA_pred_resp,
                :sV_resp => sV_pred_resp,
                :sV_hat => sV_hat,
                :sA_hat => sA_hat,
                :s_hat_common => s_hat_common,
                :sV_hat_indep => sV_hat_indep,
                :sA_hat_indep => sA_hat_indep,
                :conditions => DataFrame(locV = conditions[indCond, 1], locA = conditions[indCond, 2]),
                :logLikeA => logLikeA,
                :logLikeV => logLikeV,
                :parameters => DataFrame(parameters'),
            )

            if length(responseLoc) > 1
                mdlEval[:freq_predV] = freq_predV
                mdlEval[:freq_predA] = freq_predA
            end
        end
    end

    # sum over conditions and turn into negative log likelihood
    negLogLike = -sum(logLikeCond)

    return negLogLike, mdlEval
end

end  # end of module


# Helper function to check membership in an array
function ismember(arr, val)
    return in(val, arr)
end

# Helper function to get conditions and data conditions
function get_conditions_data_conditions(dataVA)
    tol = eps(Float32)
    conditions = Set()

    # Ensure, that the order of variables in dataVA is correct, also convert
    # from table to matrix
    dataVA = convert(Matrix, dataVA[:, ["locV", "locA", "respV", "respA"]])

    # Since condition labels are floating point numbers, we have to use a
    # tolerance value for comparison operations.
    for i in 1:size(dataVA, 1)
        conditions_union = Set(consolidator(dataVA[i, [1, 2]], [], [], tol))
        conditions = union(conditions, conditions_union)
    end

    nCond = length(conditions)
    dataConditions = Vector{DataFrame}(undef, nCond)

    for (i, cond) in enumerate(conditions)
        # Finding the examples corresponding to a condition.
        match = all(abs.(dataVA[:, [1, 2]] .- cond) .< tol, dims=2)
        dataConditions[i] = DataFrame(dataVA[match, [3, 4]])
    end

    return collect(conditions), dataConditions
end
