# Borrowed from Lux.jl
fast_chunk(h::Int, n::Int) = (1:h) .+ h * (n - 1)
function fast_chunk(x::AbstractArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return selectdim(x, dim, fast_chunk(h, n))
end
function fast_chunk(x::CuArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return copy(selectdim(x, dim, fast_chunk(h, n)))
end
function fast_chunk(x::AbstractArray, ::Val{N}, d::Val{D}) where {N,D}
    return fast_chunk.((x,), size(x, D) ÷ N, 1:N, d)
end
