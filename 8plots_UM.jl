using Pkg
Pkg.activate("../EEG_plots/env")

using Unfold
using CairoMakie
using WGLMakie
using Pipe
using LinearAlgebra
using TopoPlots
using PyMNE
using StatsBase # mean/std
using UnfoldSim

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
using CategoricalArrays
using MakieThemes
using CSV
set_theme!(theme_ggthemr(:fresh))
include("example_data.jl")

Makie.inline!(true)

raw_ch_names = ["FP1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7", "P9", "PO7", "PO3", "O1", "Oz", "Pz", "CPz", "FP2", "Fz", "F4", "F8", "FC4", "FCz", "Cz", "C4", "C6", "P4", "P8", "P10", "PO8", "PO4", "O2"]

#=
begin # load  one single-subject dataset 
    p = "../EEG_plots/data/sub-002_ses-N170_task-N170_eeg.set"
    raw = PyMNE.io.read_raw_eeglab(p, preload=true)
    evt_e = DataFrame(load("data/evt_e.jld2")["1"])
    dat_e = load("data/dat_e.jld2")["1"]
    mon = PyMNE.channels.make_standard_montage("standard_1020")
    raw.set_channel_types(Dict("HEOG_left" => "eog", "HEOG_right" => "eog", "VEOG_lower" => "eog"))
    raw.set_montage(mon, match_case=false)
    pos = PyMNE.channels.make_eeg_layout(raw.info).pos
    pos = pyconvert(Array, pos)
    pos = [Point2f(pos[k, 1], pos[k, 2]) for k in 1:size(pos, 1)]
    raw_ch_names = pyconvert(Array, raw.ch_names)
end;
begin
    # times vector (from-to)
    times = range(-0.3, length=size(dat_e, 2), step=1 ./ 128)

    # get standard errors
    se_solver = (x, y) -> Unfold.solver_default(x, y, stderror=true)
    # define effect-coding
    contrasts = Dict(:category => DummyCoding(), :condition => DummyCoding())

    analysis_formula = @formula 0 ~ 1 + category * condition

    results_allSubjects = DataFrame()

    for sub ∈ unique(evt_e.subject)

        # select events of one subject
        sIx = evt_e.subject .== sub

        # fit Unfold-Model
        # declaring global so we can access the variable outside of this loop (local vs. global scope)
        global mres = Unfold.fit(UnfoldModel,
            analysis_formula,
            evt_e[sIx, :],
            dat_e[:, :, sIx],
            times,
            contrasts=contrasts,
            solver=se_solver, show_progress=false)

        # make results of one subject available
        global results_onesubject = coeftable(mres)

        # concatenate results of all subjects
        results_onesubject[!, :subject] .= sub
        append!(results_allSubjects, results_onesubject)

    end

end;
times = range(-0.3, length=size(dat_e,2), step=1 ./ 128)
data = filter(x -> x.coefname == "category: face" || x.coefname == "(Intercept)", results_allSubjects) 
data.coefname = replace(data.coefname, "category: face" => "A", "(Intercept)" => "B")
nsubject = length(unique(data[!, "subject"]))

data = @pipe data |> 
    groupby(_, :channel) |> # baseline correction using lambda function 
    transform(_, [:estimate,:time] => (x,t) -> x .- mean(x[t .< 0])) |> 
    rename!(_, :estimate_time_function => :estimate_bsln_corrected) |>
    @subset(_, :channel .== 28) |> 
    rename!(_, :coefname => :conditions) |> 
    groupby(_, [:time, :conditions]) |> 
    @transform!(_, :estimate_mean = mean(:estimate_bsln_corrected), :stderror_mean = 2*(mean(:stderror)./sqrt(nsubject)))|> 
    @subset(_, :subject .== 1)  |> 
    select(_, Not([:basisname, :group, :channel, :subject, :estimate, :estimate_bsln_corrected, :stderror])) |>
    groupby(_, :time) |>
    @transform!(_, :sum_mean = sum(:estimate_mean))|> 
    @transform!(_, :sum_band = sum(:stderror_mean))|> 
    @subset(_, :conditions .== "A") 

CSV.write("../EEG_plots/data/data_erp.csv", data)

data2 = @pipe results_onesubject |>
                 select(_, Not([:basisname, :group])) |>
                 rename!(_, :coefname => :category, :estimate => :yhat)

data2.category = recode(data2.category, "(Intercept)" => "intact car", "category: face" => "intact face",
        "condition: scrambled" => "scrambled\ncar", "category: face & condition: scrambled" => "scrambled\nface")

CSV.write("../EEG_plots/data/data_pp.csv", data2)
=#

function line_plot(f)
    data = CSV.read("../EEG_plots/data/data_erp.csv", DataFrame)
    #times = range(-0.3, length=size(data, 2), step=1 ./ 128)

    ax = Axis(f[1, 1],
        xlabel="Time [s]", ylabel="Voltage amplitude [µV]")
    hlines!(0, color=:gray, linewidth=1)
    vlines!(0, color=:gray, linewidth=1)
    band!(data.time, data.estimate_mean - data.stderror_mean, data.estimate_mean + data.stderror_mean, color=(:steelblue1, 0.5)) #colormap=:viridis)
    band!(data.time, data.sum_mean - data.stderror_mean, data.sum_mean + data.stderror_mean, color=(:goldenrod2, 0.5))

    lines!(data.time, data.estimate_mean, label="A", color=:steelblue1, linewidth=2)
    lines!(data.time, data.sum_mean, label="B", color=:goldenrod2, linewidth=3)

    xlims!(-0.3, 0.8)
    #Legend(f[1, 2], ax, "Conditions", framevisible=false)
    hidespines!(ax, :t, :r) # delete unnecessary spines (lines)
    hidedecorations!(ax, label=false, ticks=false, ticklabels=false)
    f
    #save("plots/plot1.svg", f)
end
line_plot(Figure())

function butterfly_plot(f)
    #data, pos = TopoPlots.example_data()
    ax = Axis(f[1, 1], xlabel="Time [s]", ylabel="Voltage amplitude [µV]")
    data, pos = example_data("TopoPlots.jl")
    #plot_butterfly(df)
    plot_butterfly!(f[1, 1], data; positions=pos)
    hidedecorations!(ax)
    hidespines!(ax)

    f
end
butterfly_plot(Figure())

function topo_plot(f, g=nothing)
    data, positions = TopoPlots.example_data()

    t = 100
    if isnothing(g)
        ax = Axis(f[1, 1], aspect=DataAspect())
        plot_topoplot!(f[1, 1], data[:, 340, 1]; positions=positions, visual=(label_scatter=false,))
    else
        ax = g[1, 1] = Axis(f, aspect=DataAspect())
        plot_topoplot!(g[1, 1], data[:, 340, 1]; positions=positions, visual=(label_scatter=false,))
    end


    text!(0.5, -0.3, text="[" .* string.(t) .* " ms]", align=(:center, :center))

    hidedecorations!(ax)
    hidespines!(ax)

    f
end
#topo_plot(Figure())

function topo_vector(f, g=nothing)
    if isnothing(g)
        ax = Axis(f[2, 1:5], aspect=DataAspect())
    else
        ax = g[2, 1:5] = Axis(f, aspect=DataAspect())
    end
    data, positions = TopoPlots.example_data()
    df = UnfoldMakie.eeg_matrix_to_dataframe(data[:, :, 1], string.(1:length(positions)))

    Δbin = 80
    chaLeng = 5
    x = Array(55:120:600)
    t = Array(-0.3:0.18:0.5)
    text!(x, fill(35, chaLeng), text="[" .* string.(t) .* " s]", align=(:center, :center))

    xlims!(low=0, high=600)
    ylims!(low=0, high=110)

    hidespines!(ax)
    hidedecorations!(ax, label=false)
    if isnothing(g)
        plot_topoplotseries!(f[1:2, 1:5], df, Δbin; positions=positions, visual=(label_scatter=false,))
    else
        plot_topoplotseries!(g[1:2, 1:5], df, Δbin; positions=positions, visual=(label_scatter=false,))
    end

    f
end
#topo_vector(Figure())


function topo_array(f; draw_labels=false, times=nothing)
    num = 30#64
    data, pos = TopoPlots.example_data()
    data = data[1:num, :, 1]

    times = isnothing(times) ? (1:size(data, 2)) : times

    pos = hcat([[p[1], p[2]] for p in pos]...)

    pos = pos[:, 1:num]
    minmaxrange = (maximum(pos, dims=2) - minimum(pos, dims=2))
    pos = (pos .- mean(pos, dims=2)) ./ minmaxrange .+ 0.5

    axlist = []
    #ax = Axis(f[1, 1],backgroundcolor=:green)#

    rel_zeropoint = argmin(abs.(times)) ./ length(times)

    for (ix, p) in enumerate(eachcol(pos))
        x = p[1] #- 0.1
        y = p[2] #- 0.1
        # todo: 0.1 should go into plot config
        ax = Axis(f[1, 1], width=Relative(0.2), height=Relative(0.2),
            halign=x, valign=y)# title = raw_ch_names[1:30])
        if draw_labels
            text!(ax, rel_zeropoint + 0.1, 1, color=:gray, fontsize=12, text=string.(ix), align=(:left, :top), space=:relative)
        end
        # todo: add label if not nothing

        push!(axlist, ax)
    end
    # todo: make optional + be able to specify the linewidth + color
    hlines!.(axlist, Ref([0.0]), color=:gray, linewidth=0.5)
    vlines!.(axlist, Ref([0.0]), color=:gray, linewidth=0.5)

    times = isnothing(times) ? (1:size(data, 2)) : times

    # todo: add customizable kwargs
    h = lines!.(axlist, Ref(times), eachrow(data))

    linkaxes!(axlist...)
    hidedecorations!.(axlist)
    hidespines!.(axlist)

    f

end

topo_array(Figure())


#=
evts = CSV.read("/store/data/WLFO/derivatives/preproc_agert/sub-20/eeg/sub-20_task-WLFO_events.tsv", DataFrame)
evts.latency = evts.onset .* 512
evts_fix = subset(evts, :type => x -> x .== "fixation")
raw = PyMNE.io.read_raw_eeglab("/store/data/WLFO/derivatives/preproc_agert/sub-20/eeg/sub-20_task-WLFO_eeg.set")
d, times = Unfold.epoch(pyconvert(Array, raw.get_data(units="uV")), evts_fix, (-0.1, 1), 512)
coalesce.(d[1, :, :], NaN)
f = Figure()
d_nan = coalesce.(d[1, :, :], NaN)

CSV.write("../EEG_plots/data/data_erpimage.csv", Tables.table(d_nan), writeheader=false)
CSV.write("../EEG_plots/data/evts_erpimage.csv", evts_fix)
=#
function ERPplot(f)
    data_erpimage = CSV.read("../EEG_plots/data/data_erpimage.csv", Tables.matrix, header=0)
    evts_erpimage = CSV.read("../EEG_plots/data/evts_erpimage.csv", DataFrame) 
    times = -0.099609375:0.001953125:1.0
    color_range = (; colorrange=(-10, 10))
    plot_erpimage!(f[1, 1], times, data_erpimage; sortvalues=diff(evts_fix.onset ./ 100), visual=color_range)
    f
end
#ERPplot(Figure())

function channelplot(f)
    f = Figure()
    x = [i[1] for i in pos]
    y = [i[2] for i in pos]

    x = round.(x; digits=2)
    y = Integer.(round.((y .- mean(y)) * 20)) * -1
    x = Integer.(round.((x .- mean(x)) * 20))
    d = zip(x, y, raw.ch_names, 1:20)
    a = sort!(DataFrame(d), [:2, :1], rev=[true, false])
    b = a[!, :4]
    c = a[!, :3]
    c = pyconvert(Array, c)
    c = [string(x) for x in c]

    ix = range(-0.3, 1.2, length=size(dat_e, 2))
    iy = 1:20
    iz = mean(dat_e, dims=3)[b, :, 1]'

    gin = f[1, 1] = GridLayout()
    ax = Axis(gin[1, 1], xlabel="Time [s]", ylabel="Channels")
    hm = CairoMakie.heatmap!(ix, iy, iz, # how to reshape this into matrix???
        colormap="cork") # single trial
    ax.yticks = iy
    ax.ytickformat = xc -> c
    ax.yticklabelsize = 14

    CairoMakie.Colorbar(gin[1, 2], hm, label="Voltage [µV]")
    f
    #save("plots_jpg/plot8.jpg", f)
end

#channelplot(Figure())

function par_plot(f, data, width, height, gap, plot)
    # channels data
    channels = [10, 11, 14, 28, 29, 30]
    ch = raw_ch_names[channels]
    chaLeng = length(channels)

    # get a colormap for each category
    categories = unique(data.category)
    colors = Dict{String,RGBA{Float64}}()
    catLeng = length(categories)
    bord = 2 # colormap border (prevents from using outer parts of color map)
    colormap = cgrad(:roma, (catLeng < 2) ? 2 + (bord * 2) : catLeng + (bord * 2), categorical=true) # haline
    for i in eachindex(categories)
        setindex!(colors, colormap[i+bord], categories[i])
    end

    # limits
    limits = []
    l_low = []
    l_up = []
    for cha in channels
        tmp = filter(x -> (x.channel == cha), data)
        w = extrema.([tmp.yhat])
        append!(limits, w)
        append!(l_up, w[1][2])
        append!(l_low, w[1][1])

    end

    # scalers  
    #width = 500;   height = 30 ;     

    bottom_padding = 7
    y = fill(105, chaLeng) # height of plot

    # axes
    gin = f[1, 1] = GridLayout()
    ax = Axis(gin[1, 1:4])

    for i in 1:chaLeng
        x = (i - 1) / (chaLeng - 1) * width
        Makie.LineAxis(ax.scene, limits=limits[i], # maybe consider as unique axis????
            spinecolor=:black, labelfont="Arial",
            labelrotation=0.0,
            ticklabelfont="Arial", spinevisible=true, ticklabelsvisible=false, #switch, 
            minorticks=IntervalsBetween(2),  #tickcolor = :red, 
            endpoints=Point2f[(x, bottom_padding), (x, height)],
            ticklabelalign=(:right, :center), labelvisible=false)
    end

    # line scaling
    for time in unique(data.time)
        tmp1 = filter(x -> (x.time == time), data) #1 timepoint, 10 rows (2 conditions, 5 channels) 
        for cat in categories
            # df with the order of the channels
            dfInOrder = data[[], :]
            tmp2 = filter(x -> (x.category == cat), tmp1)

            # create new dataframe with the right order
            for cha in channels
                append!(dfInOrder, filter(x -> (x.channel == cha), tmp2))
            end

            values = map(1:chaLeng, dfInOrder.yhat, limits) do q, d, l # axes, data
                z = (q - 1) / (chaLeng - 1) * width
                Point2f(z, (d - l[1]) ./ (l[2] - l[1]) * (height - bottom_padding) + bottom_padding)
            end
            lines!(ax.scene, values; color=colors[cat])
        end
    end

    # axis labels 
    ax.xlabel = "Channels"
    ax.ylabel = "Voltage amplitude [µV]"
    x = Array(15:(width-15)/(chaLeng-1):width) # the width of the plot is set, so the labels have to be placed evenly

    text!(x, y, text=ch, align=(:right, :center), # channels lables
        offset=(0, 10),
        color=:blue)
    text!(x, fill(3, chaLeng), align=(:right, :center), text=string.(round.(l_low, digits=1))) # lower limit lables
    text!(x, fill(100, chaLeng), align=(:right, :center), text=string.(round.(l_up, digits=1))) # upper limit lables

    # text legend
    Makie.xlims!(low=-40, high=500)
    Makie.ylims!(low=0, high=120)
    hidespines!(ax)
    hidedecorations!(ax, label=false)

    # legend
    ax2 = Axis(gin[1, 5:6])
    for cat in categories # helper, cuz without them they wouldn't have an entry in legend
        lines!(ax2, 1, 1, 1, label=cat, color=colors[cat])
    end

    # legend adjustment 
    #axislegend(ax2, position = :rc, framevisible = false)
    Legend(gin[1, 5:6], ax2, "Conditions", framevisible=false)
    hidespines!(ax2)
    hidedecorations!(ax2, label=false)

    colgap!(gin, gap)


    # experimental
    w = @lift widths($(ax.scene.px_area))[1]
    h = @lift widths($(ax.scene.px_area))[2]
    # println(w, " ", h) 

    if plot == true
        save("plots/plot7.jpg", f)
    end
    f
end
#data_pp = CSV.read("../EEG_plots/data/data_pp.csv", DataFrame)
par_plot(Figure(), data_pp, 493, 95, 0, false)

function comb_plot()
    f = Figure(#backgroundcolor = RGBf(0.98, 0.98, 0.98),
        resolution=(1200, 1400)
    )
    ga = f[1, 1] = GridLayout()
    gc = f[2, 1] = GridLayout()
    ge = f[3, 1] = GridLayout()
    gg = f[4, 1] = GridLayout()
    geh = f[1:4, 2] = GridLayout()
    gb = geh[1, 1] = GridLayout()
    gd = geh[2, 1] = GridLayout()
    gf = geh[3, 1] = GridLayout()
    gh = geh[4, 1] = GridLayout()

    line_plot(ga)
    df, pos = example_data("TopoPlots.jl")
    plot_butterfly!(gb[1, 1], df; positions=pos)

    topo_plot(f, gc)
    topo_vector(f, gd)
    topo_array(ge)
    ERPplot(gf)
    channelplot(gg)

    data_pp = CSV.read("../EEG_plots/data/data_pp.csv", DataFrame)
    par_plot(gh, data_pp, 493, 95, 0, false)

    for (label, layout) in zip(["A", "B", "C", "D", "E", "F", "G", "H"], [ga, gb, gc, gd, ge, gf, gg, gh])
        Label(layout[1, 1, TopLeft()], label,
            fontsize=26,
            font=:bold,
            padding=(0, 5, 5, 0),
            halign=:right)
    end
    #f
    save("plots/comb.jpg", f)
end
comb_plot()

#=
function comb_plot1()
    f = Figure(#backgroundcolor = RGBf(0.98, 0.98, 0.98),
        resolution=(1000, 1200)
    )
    ga = f[1, 1] = GridLayout()
    gc = f[2, 1] = GridLayout()
    ge = f[3, 1] = GridLayout()
    gg = f[4, 1] = GridLayout()
    geh = f[1:4, 2] = GridLayout()
    gb = geh[1, 1] = GridLayout()
    gd = geh[2, 1] = GridLayout()
    gf = geh[3, 1] = GridLayout()
    gh = geh[4, 1] = GridLayout()

    ERPplot(ga)

    for (label, layout) in zip(["A", "B", "C", "D", "E", "F", "G", "H"], [ga, gb, gc, gd, ge, gf, gg, gh])
        Label(layout[1, 1, TopLeft()], label,
            fontsize=26,
            font=:bold,
            padding=(0, 5, 5, 0),
            halign=:right)
    end
    f
   
end
comb_plot1()
=#




