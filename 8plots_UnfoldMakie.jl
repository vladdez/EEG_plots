using Pkg
Pkg.activate("../EEG_plots/env")
import Pkg; Pkg.add("CategoricalArrays")

using UnfoldMakie
using CairoMakie
using DataFramesMeta
using UnfoldSim
using Unfold
using MakieThemes
set_theme!(theme_ggthemr(:fresh)) 
using PyMNE
using PythonCall
using Unfold
using CairoMakie
using GLMakie
using Pipe
using LinearAlgebra
using TopoPlots
using PyMNE
using StatsBase # mean/std

using JLD2 # loading data
using ColorSchemes
using Colors
using DataFrames # for image
using DataFramesMeta # @subset etc. working with DataFrames
using ImageFiltering # for kernel
using StatsModels # UnfoldFit
using FileIO
using UnfoldMakie
using PythonCall


p = "../EEG_plots/data/sub-002_ses-N170_task-N170_eeg.set"
raw = PyMNE.io.read_raw_eeglab(p, preload=true)
evt_e = DataFrame(load("data/evt_e.jld2")["1"])
dat_e =  load("data/dat_e.jld2")["1"]
mon = PyMNE.channels.make_standard_montage("standard_1020")
raw.set_channel_types(Dict("HEOG_left"=>"eog","HEOG_right"=>"eog","VEOG_lower"=>"eog"))
raw.set_montage(mon,match_case=false)
pos = PyMNE.channels.make_eeg_layout(raw.info).pos
pos = pyconvert(Array,pos) 
pos = [Point2f(pos[k,1], pos[k,2]) for k in 1:size(pos,1)]



include("example_data.jl")





data, positions = TopoPlots.example_data()
df = UnfoldMakie.eeg_matrix_to_dataframe(data[:,:,1], string.(1:length(positions)));
df2 = insertcols!(df, 4, :channel =>  df[!, :label])

Δbin = 80
plot_topoplotseries(df2, Δbin; positions=positions)
