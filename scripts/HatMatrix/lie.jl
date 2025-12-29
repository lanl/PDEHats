## Translations
const L_SHIFT = 127
for (dx, dy) in Iterators.product((-L_SHIFT):L_SHIFT, (-L_SHIFT):L_SHIFT)
    if (dx == 0) && (dy == 0)
        fname = Symbol("g_identity")
    else
        fname = Symbol("g_shift_x_$(dx)_y_$(dy)")
    end
    @eval function $(fname)(trajectory::AbstractArray{Float32,5})
        return circshift(trajectory, ($(dx), $(dy), 0, 0, 0))
    end
end
function get_translations_line_x(; L::Int=127)
    dy = 0
    translations = map(1:L) do dx
        fname = Symbol("g_shift_x_$(dx)_y_$(dy)")
        @eval return $(fname)
    end
    return vec(translations)
end
function get_translations_line_y(; L::Int=127)
    dx = 0
    translations = map(1:L) do dy
        fname = Symbol("g_shift_x_$(dx)_y_$(dy)")
        @eval return $(fname)
    end
    return vec(translations)
end
function get_translations_line(; L::Int=127)
    translations_x = get_translations_line_x(; L=L)
    translations_y = get_translations_line_y(; L=L)
    translations = [translations_x..., translations_y...]
    return translations
end
function get_translations_box(; L::Int=17)
    translations = map(Iterators.product(1:L, 1:L)) do (dx, dy)
        fname = Symbol("g_shift_x_$(dx)_y_$(dy)")
        @eval return $(fname)
    end
    return vec(translations)
end
## Rotations
function g_rotate_90(trajectory::AbstractArray{Float32,5})
    F = size(trajectory, 3)
    if F == 4
        trajectory_rotated_90 = g_rotate_90_CE(trajectory)
    elseif F == 2
        trajectory_rotated_90 = g_rotate_90_NS(trajectory)
    end
    return trajectory_rotated_90
end
function g_rotate_90_CE(trajectory::AbstractArray{Float32,5})
    (Lx, Ly, F, T, B) = size(trajectory)
    ##
    density_rotated = reshape(
        stack(
            map(Iterators.product(1:T, 1:B)) do (t, b)
                img = trajectory[:, :, 1, t, b]
                img_rotated = rotl90(img)
                return img_rotated
            end,
        ), (Lx, Ly, 1, T, B)
    )
    mom_x_rotated = reshape(
        stack(
            map(Iterators.product(1:T, 1:B)) do (t, b)
                img = -trajectory[:, :, 3, t, b]
                img_rotated = rotl90(img)
                return img_rotated
            end,
        ), (Lx, Ly, 1, T, B)
    )
    mom_y_rotated = reshape(
        stack(
            map(Iterators.product(1:T, 1:B)) do (t, b)
                img = trajectory[:, :, 2, t, b]
                img_rotated = rotl90(img)
                return img_rotated
            end,
        ), (Lx, Ly, 1, T, B)
    )
    energy_rotated = reshape(
        stack(
            map(Iterators.product(1:T, 1:B)) do (t, b)
                img = trajectory[:, :, 4, t, b]
                img_rotated = rotl90(img)
                return img_rotated
            end,
        ), (Lx, Ly, 1, T, B)
    )
    ##
    trajectory_rotated_90 = cat(
        density_rotated, mom_x_rotated, mom_y_rotated, energy_rotated; dims=3
    )
    return trajectory_rotated_90
end
function g_rotate_90_NS(trajectory::AbstractArray{Float32,5})
    (Lx, Ly, F, T, B) = size(trajectory)
    ##
    vel_x_rotated = reshape(
        stack(
            map(Iterators.product(1:T, 1:B)) do (t, b)
                img = -trajectory[:, :, 2, t, b]
                img_rotated = rotl90(img)
                return img_rotated
            end,
        ), (Lx, Ly, 1, T, B)
    )
    vel_y_rotated = reshape(
        stack(
            map(Iterators.product(1:T, 1:B)) do (t, b)
                img = trajectory[:, :, 1, t, b]
                img_rotated = rotl90(img)
                return img_rotated
            end,
        ), (Lx, Ly, 1, T, B)
    )
    ##
    trajectory_rotated_90 = cat(vel_x_rotated, vel_y_rotated; dims=3)
    return trajectory_rotated_90
end
function g_rotate_180(trajectory::AbstractArray{Float32,5})
    trajectory_rotated_90 = g_rotate_90(trajectory)
    trajectory_rotated_180 = g_rotate_90(trajectory_rotated_90)
    return trajectory_rotated_180
end
function g_rotate_270(trajectory::AbstractArray{Float32,5})
    trajectory_rotated_90 = g_rotate_90(trajectory)
    trajectory_rotated_180 = g_rotate_90(trajectory_rotated_90)
    trajectory_rotated_270 = g_rotate_90(trajectory_rotated_180)
    return trajectory_rotated_270
end
function g_flip(trajectory::AbstractArray{Float32,5})
    F = size(trajectory, 3)
    if F == 4
        trajectory_flipped = g_flip_CE(trajectory)
    elseif F == 2
        trajectory_flipped = g_flip_NS(trajectory)
    end
    return trajectory_flipped
end
function g_flip_CE(trajectory::AbstractArray{Float32,5})
    flip_vec_r = [1, -1, 1, 1]
    flip_vec = reshape(flip_vec_r, (1, 1, 4, 1, 1))
    trajectory_reversed = reverse(trajectory; dims=1)
    trajectory_flipped = flip_vec .* trajectory_reversed
    return trajectory_flipped
end
function g_flip_NS(trajectory::AbstractArray{Float32,5})
    flip_vec_r = [-1, 1]
    flip_vec = reshape(flip_vec_r, (1, 1, 2, 1, 1))
    trajectory_reversed = reverse(trajectory; dims=1)
    trajectory_flipped = flip_vec .* trajectory_reversed
    return trajectory_flipped
end
function g_flip_rotate_90(trajectory::AbstractArray{Float32,5})
    trajectory_rotated_90 = g_rotate_90(trajectory)
    trajectory_flip_rotated_90 = g_flip(trajectory_rotated_90)
    return trajectory_flip_rotated_90
end
function g_flip_rotate_180(trajectory::AbstractArray{Float32,5})
    trajectory_rotated_90 = g_rotate_90(trajectory)
    trajectory_rotated_180 = g_rotate_90(trajectory_rotated_90)
    trajectory_flip_rotated_180 = g_flip(trajectory_rotated_180)
    return trajectory_flip_rotated_180
end
function g_flip_rotate_270(trajectory::AbstractArray{Float32,5})
    trajectory_rotated_90 = g_rotate_90(trajectory)
    trajectory_rotated_180 = g_rotate_90(trajectory_rotated_90)
    trajectory_rotated_270 = g_rotate_90(trajectory_rotated_180)
    trajectory_flip_rotated_270 = g_flip(trajectory_rotated_270)
    return trajectory_flip_rotated_270
end
function get_rotations()
    rotations = [
        g_identity,
        g_rotate_90,
        g_rotate_180,
        g_rotate_270,
        g_flip,
        g_flip_rotate_90,
        g_flip_rotate_180,
        g_flip_rotate_270,
    ]
    return rotations
end
