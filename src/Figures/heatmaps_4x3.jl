##
function animate_heatmaps_4x3(
    trajectory_1::AbstractArray{Float32,4},
    trajectory_2::AbstractArray{Float32,4},
    trajectory_3::AbstractArray{Float32,4},
    supertitle::AbstractString,
    titles::NTuple{3,AbstractString},
    labels_y::NTuple{4,AbstractString};
    size_supertitle::Int=20,
    size_title::Int=12,
    size_ylabel::Int=12,
    path_save::String=projectdir("dir_save_default/animate_heatmaps_4x3"),
    framerate::Int=3,
    size_fig::NTuple{2,Int}=(310, 400),
    padding_figure::NTuple{4,Int}=(1, 1, 20, 1),
    label_colorbar::Union{Nothing,AbstractString}=nothing,
    size_label_colorbar::Int=10,
    colormap=:balance,
)
    ## Range: Min
    min_1 = minimum(trajectory_1; dims=(1, 2, 4))
    min_2 = minimum(trajectory_2; dims=(1, 2, 4))
    min_3 = minimum(trajectory_3; dims=(1, 2, 4))
    ranges_color_min = vec(minimum(cat(min_1, min_2, min_3; dims=1); dims=1))
    ## Range: Max
    max_1 = maximum(trajectory_1; dims=(1, 2, 4))
    max_2 = maximum(trajectory_2; dims=(1, 2, 4))
    max_3 = maximum(trajectory_3; dims=(1, 2, 4))
    ranges_color_max = vec(maximum(cat(max_1, max_2, max_3; dims=1); dims=1))
    ##
    range_color_rows = map(
        i -> (ranges_color_min[i], ranges_color_max[i]), 1:size(trajectory_1, 3)
    )
    times = 1:size(trajectory_1, 4)
    time_Obs = Observable(times[1])
    ## Density
    density_1 = @lift(trajectory_1[:, :, 1, $time_Obs])
    density_2 = @lift(trajectory_2[:, :, 1, $time_Obs])
    density_3 = @lift(trajectory_3[:, :, 1, $time_Obs])
    ## Momentum-x
    momentum_x_1 = @lift(trajectory_1[:, :, 2, $time_Obs])
    momentum_x_2 = @lift(trajectory_2[:, :, 2, $time_Obs])
    momentum_x_3 = @lift(trajectory_3[:, :, 2, $time_Obs])
    ## Momentum-y
    momentum_y_1 = @lift(trajectory_1[:, :, 3, $time_Obs])
    momentum_y_2 = @lift(trajectory_2[:, :, 3, $time_Obs])
    momentum_y_3 = @lift(trajectory_3[:, :, 3, $time_Obs])
    ## Energy
    energy_1 = @lift(trajectory_1[:, :, 4, $time_Obs])
    energy_2 = @lift(trajectory_2[:, :, 4, $time_Obs])
    energy_3 = @lift(trajectory_3[:, :, 4, $time_Obs])
    ##
    fig = with_theme(theme_aps_2col(); figure_padding=padding_figure) do
        fig = Figure(; size=size_fig)
        supertitle = Label(fig[0, 1:3], supertitle; fontsize=size_supertitle)
        # Density
        ax_11 = Makie.Axis(
            fig[1, 1];
            title=titles[1],
            titlesize=size_title,
            ylabel=labels_y[1],
            ylabelsize=size_ylabel,
            aspect=DataAspect(),
        )
        ax_12 = Makie.Axis(
            fig[1, 2];
            title=titles[2],
            titlesize=size_title,
            aspect=DataAspect(),
        )
        ax_13 = Makie.Axis(
            fig[1, 3];
            title=titles[3],
            titlesize=size_title,
            aspect=DataAspect(),
        )
        #
        hidedecorations!(ax_11; label=false)
        hidedecorations!(ax_12; label=false)
        hidedecorations!(ax_13; label=false)
        #
        hm_11 = heatmap!(
            ax_11,
            density_1;
            colorrange=range_color_rows[1],
            colormap=colormap,
        )
        hm_12 = heatmap!(
            ax_12,
            density_2;
            colorrange=range_color_rows[1],
            colormap=colormap,
        )
        hm_13 = heatmap!(
            ax_13,
            density_3;
            colorrange=range_color_rows[1],
            colormap=colormap,
        )
        #
        cb_1 = Colorbar(
            fig[1, 4]; colorrange=range_color_rows[1], colormap=colormap
        )
        # Velocity-x
        ax_21 = Makie.Axis(
            fig[2, 1];
            ylabel=labels_y[2],
            ylabelsize=size_ylabel,
            aspect=DataAspect(),
        )
        ax_22 = Makie.Axis(fig[2, 2]; aspect=DataAspect())
        ax_23 = Makie.Axis(fig[2, 3]; aspect=DataAspect())
        #
        hidedecorations!(ax_21; label=false)
        hidedecorations!(ax_22; label=false)
        hidedecorations!(ax_23; label=false)
        #
        hm_21 = heatmap!(
            ax_21,
            momentum_x_1;
            colorrange=range_color_rows[2],
            colormap=colormap,
        )
        hm_22 = heatmap!(
            ax_22,
            momentum_x_2;
            colorrange=range_color_rows[2],
            colormap=colormap,
        )
        hm_23 = heatmap!(
            ax_23,
            momentum_x_3;
            colorrange=range_color_rows[2],
            colormap=colormap,
        )
        #
        cb_2 = Colorbar(
            fig[2, 4]; colorrange=range_color_rows[2], colormap=colormap
        )
        # Velocity-y
        ax_31 = Makie.Axis(
            fig[3, 1];
            ylabel=labels_y[3],
            ylabelsize=size_ylabel,
            aspect=DataAspect(),
        )
        ax_32 = Makie.Axis(fig[3, 2]; aspect=DataAspect())
        ax_33 = Makie.Axis(fig[3, 3]; aspect=DataAspect())
        #
        hidedecorations!(ax_31; label=false)
        hidedecorations!(ax_32; label=false)
        hidedecorations!(ax_33; label=false)
        #
        hm_31 = heatmap!(
            ax_31,
            momentum_y_1;
            colorrange=range_color_rows[3],
            colormap=colormap,
        )
        hm_32 = heatmap!(
            ax_32,
            momentum_y_2;
            colorrange=range_color_rows[3],
            colormap=colormap,
        )
        hm_33 = heatmap!(
            ax_33,
            momentum_y_3;
            colorrange=range_color_rows[3],
            colormap=colormap,
        )
        #
        cb_3 = Colorbar(
            fig[3, 4]; colorrange=range_color_rows[3], colormap=colormap
        )
        # Energy
        ax_41 = Makie.Axis(
            fig[4, 1];
            ylabel=labels_y[4],
            ylabelsize=size_ylabel,
            aspect=DataAspect(),
        )
        ax_42 = Makie.Axis(fig[4, 2]; aspect=DataAspect())
        ax_43 = Makie.Axis(fig[4, 3]; aspect=DataAspect())
        #
        hidedecorations!(ax_41; label=false)
        hidedecorations!(ax_42; label=false)
        hidedecorations!(ax_43; label=false)
        #
        hm_41 = heatmap!(
            ax_41,
            energy_1;
            colorrange=range_color_rows[4],
            colormap=colormap,
        )
        hm_42 = heatmap!(
            ax_42,
            energy_2;
            colorrange=range_color_rows[4],
            colormap=colormap,
        )
        hm_43 = heatmap!(
            ax_43,
            energy_3;
            colorrange=range_color_rows[4],
            colormap=colormap,
        )
        #
        cb_4 = Colorbar(
            fig[4, 4]; colorrange=range_color_rows[4], colormap=colormap
        )
        #
        if label_colorbar != nothing
            cb_1.label = label_colorbar
            cb_1.labelsize = size_label_colorbar
            cb_2.label = label_colorbar
            cb_2.labelsize = size_label_colorbar
            cb_3.label = label_colorbar
            cb_3.labelsize = size_label_colorbar
            cb_4.label = label_colorbar
            cb_4.labelsize = size_label_colorbar
        end
        #
        colsize!(fig.layout, 1, Aspect(1, 1))
        colsize!(fig.layout, 2, Aspect(1, 1))
        colsize!(fig.layout, 3, Aspect(1, 1))
        #
        return current_figure()
    end
    ##
    mkpath(projectdir(dirname(path_save)))
    CairoMakie.record(
        fig, projectdir(path_save * ".mp4"), times; framerate=framerate
    ) do t
        return time_Obs[] = t
    end
    ##
    return nothing
end
