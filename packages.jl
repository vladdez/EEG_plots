# just type
# julia .\packages.jl
# also do
# python -m pip install mne, pymatreader

using Pkg

dependencies = [


     "Unfold",
     "GLMakie",
     "PyMNE",# MNE is a python library for EEG data analysis
     "AlgebraOfGraphics",# plotting Grammar of Graphics
     "CSV",
     "DataFrames",
     "StatsBase",# mean/std
     "FileIO",# loading data
     "JLD2",# loading data
     "StatsModels",# UnfoldFit
     "CairoMakie",# Plotting Backend (SVGs/PNGs)
     "Printf",# interpolate strings
     "DataFramesMeta",# @subset etc. working with DataFrames
     "StatsPlots",
     "Pipe",
     "UnfoldMakie",
     "ColorSchemes",
     "PyCall",
     "ImageFiltering",
     "GR",
     "Conda"
]

Pkg.add(dependencies)

forbuild = [
    "PyCall",
    "GR",
]
forbuild = [
Pkg.build(forbuild)