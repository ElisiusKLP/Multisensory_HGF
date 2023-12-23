"""
MCMC Diagnostics
Using Arviz for Julia API
"""

using ArviZ, ArviZPythonPlots
using Turing
using Serialization
using DataFrames, CSV, TimeSeries, Serialization
using Distributions
using Plots, StatsPlots
using ActionModels, HierarchicalGaussianFiltering
using Turing
using CategoricalArrays
using Distributed


file = "/work/Multisensory_HGF/chain_saved/bci_fit4_exp1_23-12-23.jls"

part_chains = deserialize("/work/Multisensory_HGF/chain_saved/bci_fit4_exp1_23-12-23.jls")

xy = part_chains[String31("participant 1.13")]

typeof(xy)

arviz_data = convert_to_inference_data(part_chains)

plot_autocorr(xy; var_names=["sigV", "sigA"])

deserialize(path)

