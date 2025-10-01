## Forward finite time difference
function diff_time(Q_history::AbstractArray{R,5}; T::Int=21) where {R}
    dt = R(1 / (T - 1))
    dt_Q_history = diff(Q_history; dims=4) ./ dt
    return dt_Q_history
end
## Central difference finite space divergence
function div_space(
    Q_history::AbstractArray{R,5};
    kernel_div_space::AbstractArray=get_kernel_div_space(),
) where {R}
    ##
    (Lx, Ly, chs, T, B) = size(Q_history)
    @assert chs == 2
    #
    Q_history_r = reshape(Q_history, (Lx, Ly, chs, T * B))
    Q_history_r_padded = pad_circular(Q_history_r, (2, 2, 2, 2); dims=(1, 2))
    #
    div_Q_history_r = conv(Q_history_r_padded, kernel_div_space)
    div_Q_history = reshape(div_Q_history_r, (Lx, Ly, 1, T, B))
    ##
    return div_Q_history
end
function get_kernel_div_space()
    kernel_x = get_kernel_x()
    kernel_y = get_kernel_y()
    kernel = cat(kernel_x, kernel_y; dims=3)
    return kernel
end
function div_space(
    Q_history::AbstractArray{R,6};
    kernel_div_space::AbstractArray=get_kernel_div_space(),
) where {R}
    ##
    (Lx, Ly, chs_i, chs_j, T, B) = size(Q_history)
    @assert chs_i == chs_j == 2
    ##
    Q_history_r = reshape(Q_history, (Lx, Ly, chs_i, chs_j, T * B))
    div_Q_history_r = div_space(Q_history_r; kernel_div_space=kernel_div_space)
    div_Q_history = reshape(div_Q_history_r, (Lx, Ly, chs_j, T, B))
    ##
    return div_Q_history
end
## Central difference finite space gradient
function grad_space(
    Q_history::AbstractArray{R,5};
    kernel_grad_space::AbstractArray=get_kernel_grad_space(),
) where {R}
    ##
    (Lx, Ly, chs, T, B) = size(Q_history)
    @assert chs == 1
    #
    Q_history_r = reshape(Q_history, (Lx, Ly, chs, T * B))
    Q_history_r_padded = pad_circular(Q_history_r, (2, 2, 2, 2); dims=(1, 2))
    grad_Q_history_r = conv(Q_history_r_padded, kernel_grad_space)
    grad_Q_history = reshape(grad_Q_history_r, (Lx, Ly, 2, T, B))
    ##
    return grad_Q_history
end
function get_kernel_grad_space()
    kernel_x = get_kernel_x()
    kernel_y = get_kernel_y()
    kernel = cat(kernel_x, kernel_y; dims=4)
    return kernel
end
##
function get_kernel_y(; L::Int=128)
    dx = 1 / (L - 1)
    kernel_y =
        reshape(
            [0 0 0 0 0; 0 0 0 0 0; -1 8 0 -8 1; 0 0 0 0 0; 0 0 0 0 0],
            (5, 5, 1, 1),
        ) ./ (12 * dx)
    return Float32.(kernel_y)
end
function get_kernel_x(; L::Int=128)
    dy = 1 / (L - 1)
    kernel_x =
        reshape(
            [0 0 -1 0 0; 0 0 8 0 0; 0 0 0 0 0; 0 0 -8 0 0; 0 0 1 0 0],
            (5, 5, 1, 1),
        ) ./ (12 * dy)
    return Float32.(kernel_x)
end
##
