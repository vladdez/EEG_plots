### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 4c7907ad-bc25-4188-ade7-fa69e6fc719d
# ╠═╡ show_logs = false
begin
		using Pkg # cant use internal package manager because we need the topoplot version of UnfoldMakie
		Pkg.activate("F:/Prgramming/Uni GIT/FaPraExecution")
		# Pkg.add("ColorSchemes")
		Pkg.resolve()
end

# ╔═╡ d609f3e3-ff6a-4a96-a359-e44dba93c0e0
begin
	using Revise
	using UnfoldMakie
end

# ╔═╡ f3f93d30-d2b6-11ec-3ba2-898080a75c3f
begin
	using Unfold
	using PyMNE # MNE is a python library for EEG data analysis
	using AlgebraOfGraphics # plotting Grammar of Graphics
	using CSV
	using DataFrames
	using  StatsBase # mean/std
	using FileIO # loading data
	using JLD2 # loading data
	using StatsModels # UnfoldFit
	using CairoMakie # Plotting Backend (SVGs/PNGs)
	using Printf # interpolate strings
	using DataFramesMeta # @subset etc. working with DataFrames
	using CategoricalArrays
	using Colors
	
	using TopoPlots
	using RecipesBase
end

# ╔═╡ d6119836-fc49-4731-82a6-66d25963ba2c
begin # load  one single-subject dataset
	
	# once artifacts are working, we could ship it. But for now you have to download it and 
 #p =joinpath(pathof(UnfoldMakie),"artifacts","sub002_ses-N170_task-N170_eeg.set")
	    
	p = "F:\\Prgramming\\Uni GIT\\FaPraExecution\\sub-002_ses-N170_task-N170_eeg.set"
    raw = PyMNE.io.read_raw_eeglab(p,preload=false)
end;

# ╔═╡ c6bb6bbb-55a4-4bfd-bb12-24307607248a
begin # load unfold-fitted dataset of all subjects
	# takes ~25s to load because it is quite large :)
	p_all = "F:\\Prgramming\\Uni GIT\\FaPraExecution\\erpcore-N170.jld2"
	presaved_data = load(p_all)
	dat_e = presaved_data["data_e_all"].* 1e6
	evt_e = presaved_data["df_e_all"]
end;

# ╔═╡ 434a957c-ec3f-4496-a7e6-a0ec4b431754
# ╠═╡ show_logs = false
begin
	# times vector (from-to)
	times = range(-0.3, length=size(dat_e,2), step=1 ./ 128)

	# get standard errors
	se_solver =(x,y)->Unfold.solver_default(x,y,stderror=true)
	# define effect-coding
	contrasts= Dict(:category => EffectsCoding(), :condition => EffectsCoding())
	
	analysis_formula = @formula 0 ~ 1 + category * condition
	
	results_allSubjects = DataFrame()
	
	for sub ∈ unique(evt_e.subject)

		# select events of one subject
	    sIx = evt_e.subject .== sub

		# fit Unfold-Model
		# declaring global so we can access the variable outside of this loop (local vs. global scope)
	    global mres = Unfold.fit(UnfoldModel, 
						analysis_formula, 
						evt_e[sIx,:], 
						dat_e[:,:,sIx], 
						times, 
						contrasts=contrasts,
						solver=se_solver);

		# make results of one subject available
		global results_onesubject = coeftable(mres)

		# concatenate results of all subjects
	    results_onesubject[!,:subject] .= sub
	    append!(results_allSubjects,results_onesubject)
	end
	
end

# ╔═╡ 116a6d4a-248f-407f-b154-46852be66371
erp_data = dat_e[:,:,evt_e.subject .==1]

# ╔═╡ 4df1bfc8-7988-4290-b909-597d068d003c
let
	#erpimage => input als matrix
	# time x trials
	using Random
	f = Figure()
	sort_x = [[a[1] for a in argmax(erp_data[1,:,:],dims=2)]...]
	@show typeof(sort_x)
	image(f[1:4,1],erp_data[1,:,sort_x])
	lines(f[5,1],mean(erp_data[1,:,:],dims=2)[:,1])
	f
end

# ╔═╡ a3df35c3-39b5-46bb-825c-ad07e7d0ea79
# channelimage
# =>  input über DataFrame in unfold
image(mean(erp_data[1:30,:,:],dims=3)[:,:,1]')

# ╔═╡ 343f3c15-7f48-4e25-84d5-f4ef01d35db7
md"""## Designmatrix"""

# ╔═╡ 7cbe5fcb-4440-4a96-beca-733ddcb07861
# ╠═╡ disabled = true
#=╠═╡
begin
	designmatrix(mres)
	cDesign = PlotConfig(:designmatrix)
	cDesign.setExtraValues(;showLegend=true)
	cDesign.setLegendValues(;flipaxis=true, ticks=[-0.9,1])
	plot_designTest(designmatrix(mres), cDesign;sort=true)
end
  ╠═╡ =#

# ╔═╡ d329586a-fd05-4729-a0c3-4700ee22a498
md"""## Butterly Plot"""

# ╔═╡ 975382de-5f24-4e9e-87f6-a7f7ba93ff8e
begin # let (against begin) makes a local environment, not sharing the modifications / results wiht outside scripts. Beware of field assignments x.blub = "1,2,3" will overwrite x.blub, x = [1,2,3] will not overwrite global x, but make a copy
	
results_plot_butter = @subset(results_onesubject,:coefname .== "(Intercept)",:channel .<4)
	
    cButter = PlotConfig(:butterfly)
	
	cButter.setExtraValues(categoricalColor=false,
		categoricalGroup=true,
		legendPosition=:right,
		border=false,
		topoLabel=:position)
	
    # cButter.setLegendValues()
	# cButter.setColorbarValues()
	# cButter.setMappingValues(color=:channel, group=:channel)

	# for testing add a column with labels
	results_plot_butter.position = results_plot_butter[:, :channel] .|> c -> ("C" * string(c))

	
	plot_line(results_plot_butter, cButter)

end

# ╔═╡ 62632f86-5792-49c3-a229-376211deae64
md"""
## Line Plot 2
"""

# ╔═╡ 41e1b735-4c19-4e6d-8681-ef0f893aec87
let    
    results_plot = @subset(results_onesubject,:coefname .== "(Intercept)",:channel .<5)
    cLine = PlotConfig(:lineplot)
	cLine.setExtraValues(showLegend=true, legendPosition=:right, categoricalColor=false,categoricalGroup=true)
	cLine.setMappingValues(color=:channel, group=:channel)
    cLine.setLegendValues(nbanks=1)
    plot_line(results_plot, cLine)
	# results_plot
end

# ╔═╡ 98026af7-d875-43f3-a3cb-3109faef5822
md"""
## Topoplots
"""

# ╔═╡ 3cee30ae-cf25-4684-bd49-c64b0b96b4e6
# ╠═╡ disabled = true
#=╠═╡
begin
	mon = PyMNE.channels.make_standard_montage("standard_1020")
	raw.set_channel_types(Dict("HEOG_left"=>"eog","HEOG_right"=>"eog","VEOG_lower"=>"eog"))
	raw.set_montage(mon,match_case=false)
	pos = PyMNE.channels.make_eeg_layout(get_info(raw)).pos
end;
  ╠═╡ =#

# ╔═╡ 5f0f2708-f2d9-4a72-a528-185837a43e06
# ╠═╡ disabled = true
#=╠═╡
# potentially still buggy: The sensor-positions are flipped 90°
plot_topoplot(@subset(results_onesubject,:coefname.=="(Intercept)",:channel .<=30),0.2,topoplotCfg=(positions=collect(pos),))
# results_onesubject
  ╠═╡ =#

# ╔═╡ 6ee180df-4340-45a2-bb1e-37aad7953875
# ╠═╡ disabled = true
#=╠═╡
# maybe this should be the default
# note the bad time on top :S
plot_topoplot_series(@subset(results_onesubject,:coefname.=="(Intercept)",:channel .<=30),0.2,topoplotCfg=(sensors=false,positions=collect(pos[:,[2,1]]),),mappingCfg=(col=:time,))
  ╠═╡ =#

# ╔═╡ f0f752d8-7f2d-4a49-8763-53df2cff6126
# ╠═╡ disabled = true
#=╠═╡
# multi-coeffiecients dont work, because the aggregation is on groupby (channel)
plot_topoplot_series(@subset(results_onesubject,:channel .<=30),0.2,topoplotCfg=(sensors=false,positions=collect(pos[:,[2,1]]),),mappingCfg=(col=:time,row=:coefficient))
  ╠═╡ =#

# ╔═╡ 7da4df51-589a-4eb5-8f1f-f77ab65cf10a
# ╠═╡ disabled = true
#=╠═╡
@subset(results_onesubject,:coefname.=="(Intercept)")
  ╠═╡ =#

# ╔═╡ f7dc2c10-1b1e-4081-9f68-107be290d797
axisSettings = (topspinevisible=false,rightspinevisible=false,bottomspinevisible=false,leftspinevisible=false,xgridvisible=false,ygridvisible=false,xticklabelsvisible=false,yticklabelsvisible=false, xticksvisible=false, yticksvisible=false)

# ╔═╡ 6262fba5-d52a-456a-849e-1670f145caef
begin
	f = Figure()
	
	data, positions = TopoPlots.example_data()
	show(data)
	labels = ["s$i" for i in 1:size(data, 1)]
	# data[:, 340, 1]
	f, ax, h = eeg_topoplot(data[:, 340, 1], labels; label_text=false,positions=positions, axis=axisSettings)
	# show(ax.bbox)
	axis = Axis(f, bbox = BBox(100, 0, 0, 100); axisSettings...)

	
	draw = eeg_topoplot!(axis, zeros(64), labels; label_text=false,positions=positions)
	
	f
end

# ╔═╡ 51b58cdd-0cec-41d9-b778-c4b31089bcc1
begin
    
    struct NullInterpolator <: TopoPlots.Interpolator
        
    end
    
    function (ni::NullInterpolator)(
            xrange::LinRange, yrange::LinRange,
            positions::AbstractVector{<: Point{2}}, data::AbstractVector{<:Number})
    
      
        return zeros(length(xrange),length(yrange))
    end

    using ColorSchemes

    # colorscheme where first entry is 0, and exactly length(positions)+1 entries
    specialColors = ColorScheme(vcat(RGB(1,1,1.),[RGB{Float64}(i, i, i) for i in range(0,1,length(positions))]...))

    
    eeg_topoplot(1:length(positions), # go from 1:npos
        string.(1:length(positions)); 
    positions=positions,
    interpolation=NullInterpolator(), # inteprolator that returns only 0
    colorrange = (0,length(positions)), # add the 0 for the white-first color
    colormap= specialColors)
end

# ╔═╡ aaab65cc-b7d7-4949-8fd8-7bc2d4cf62f2
help_attributes(Lines)

# ╔═╡ Cell order:
# ╠═4c7907ad-bc25-4188-ade7-fa69e6fc719d
# ╠═d609f3e3-ff6a-4a96-a359-e44dba93c0e0
# ╠═f3f93d30-d2b6-11ec-3ba2-898080a75c3f
# ╠═d6119836-fc49-4731-82a6-66d25963ba2c
# ╠═c6bb6bbb-55a4-4bfd-bb12-24307607248a
# ╠═434a957c-ec3f-4496-a7e6-a0ec4b431754
# ╠═116a6d4a-248f-407f-b154-46852be66371
# ╠═4df1bfc8-7988-4290-b909-597d068d003c
# ╠═a3df35c3-39b5-46bb-825c-ad07e7d0ea79
# ╟─343f3c15-7f48-4e25-84d5-f4ef01d35db7
# ╠═7cbe5fcb-4440-4a96-beca-733ddcb07861
# ╟─d329586a-fd05-4729-a0c3-4700ee22a498
# ╠═975382de-5f24-4e9e-87f6-a7f7ba93ff8e
# ╟─62632f86-5792-49c3-a229-376211deae64
# ╠═41e1b735-4c19-4e6d-8681-ef0f893aec87
# ╟─98026af7-d875-43f3-a3cb-3109faef5822
# ╠═3cee30ae-cf25-4684-bd49-c64b0b96b4e6
# ╠═5f0f2708-f2d9-4a72-a528-185837a43e06
# ╠═6ee180df-4340-45a2-bb1e-37aad7953875
# ╠═f0f752d8-7f2d-4a49-8763-53df2cff6126
# ╠═7da4df51-589a-4eb5-8f1f-f77ab65cf10a
# ╠═f7dc2c10-1b1e-4081-9f68-107be290d797
# ╠═6262fba5-d52a-456a-849e-1670f145caef
# ╠═aaab65cc-b7d7-4949-8fd8-7bc2d4cf62f2
# ╠═51b58cdd-0cec-41d9-b778-c4b31089bcc1
