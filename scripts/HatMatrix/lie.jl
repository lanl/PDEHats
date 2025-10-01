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
function get_translations(; L::Int=8)
    translations = map(Iterators.product((-L):L, (-L):L)) do (dx, dy)
        if (dx == 0) && (dy == 0)
            fname = Symbol("g_identity")
        else
            fname = Symbol("g_shift_x_$(dx)_y_$(dy)")
        end
        @eval return $(fname)
    end
    return vec(translations)
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
## Rotations
function g_rotate_90(trajectory::AbstractArray{Float32,5})
    trajectory_rotated = stack(
        map(Iterators.product(axes(trajectory)[3:5]...)) do (f, t, b)
            img = trajectory[:, :, f, t, b]
            img_rotated = rotl90(img)
            return img_rotated
        end,
    )
    return trajectory_rotated
end
function g_rotate_180(trajectory::AbstractArray{Float32,5})
    trajectory_rotated = stack(
        map(Iterators.product(axes(trajectory)[3:5]...)) do (f, t, b)
            img = trajectory[:, :, f, t, b]
            img_rotated = rot180(img)
            return img_rotated
        end,
    )
    return trajectory_rotated
end
function g_rotate_270(trajectory::AbstractArray{Float32,5})
    trajectory_rotated = stack(
        map(Iterators.product(axes(trajectory)[3:5]...)) do (f, t, b)
            img = trajectory[:, :, f, t, b]
            img_rotated = rotr90(img)
            return img_rotated
        end,
    )
    return trajectory_rotated
end
function g_flip(trajectory::AbstractArray{Float32,5})
    _trajectory_rotated = reverse(trajectory; dims=1)
    _flip_vec = [1, -1, 1, 1]
    flip_vec = reshape(_flip_vec, (1, 1, 4, 1, 1))
    trajectory_rotated = flip_vec .* _trajectory_rotated
    return trajectory_rotated
end
function g_flip_rotate_90(trajectory::AbstractArray{Float32,5})
    _trajectory_rotated = stack(
        map(Iterators.product(axes(trajectory)[3:5]...)) do (f, t, b)
            img = trajectory[:, :, f, t, b]
            img_rotated = reverse(rotl90(img); dims=1)
            return img_rotated
        end,
    )
    _flip_vec = [1, -1, 1, 1]
    flip_vec = reshape(_flip_vec, (1, 1, 4, 1, 1))
    trajectory_rotated = flip_vec .* _trajectory_rotated
    return trajectory_rotated
end
function g_flip_rotate_180(trajectory::AbstractArray{Float32,5})
    _trajectory_rotated = stack(
        map(Iterators.product(axes(trajectory)[3:5]...)) do (f, t, b)
            img = trajectory[:, :, f, t, b]
            img_rotated = reverse(rot180(img); dims=1)
            return img_rotated
        end,
    )
    _flip_vec = [1, -1, 1, 1]
    flip_vec = reshape(_flip_vec, (1, 1, 4, 1, 1))
    trajectory_rotated = flip_vec .* _trajectory_rotated
    return trajectory_rotated
end
function g_flip_rotate_270(trajectory::AbstractArray{Float32,5})
    _trajectory_rotated = stack(
        map(Iterators.product(axes(trajectory)[3:5]...)) do (f, t, b)
            img = trajectory[:, :, f, t, b]
            img_rotated = reverse(rotr90(img); dims=1)
            return img_rotated
        end,
    )
    _flip_vec = [1, -1, 1, 1]
    flip_vec = reshape(_flip_vec, (1, 1, 4, 1, 1))
    trajectory_rotated = flip_vec .* _trajectory_rotated
    return trajectory_rotated
end
function get_rotations()
    rotations = [
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
