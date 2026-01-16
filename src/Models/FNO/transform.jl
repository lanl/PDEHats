abstract type AbstractTransform{T} end

Base.eltype(::Type{<:AbstractTransform{T}}) where {T} = T

function transform end
function truncate_modes end
function inverse end

struct FourierTransform{T,M} <: AbstractTransform{T}
    modes::M
    shift::Bool
end

function FourierTransform{T}(modes::Dims, shift::Bool=false) where {T}
    return FourierTransform{T,typeof(modes)}(modes, shift)
end

function Base.show(io::IO, ft::FourierTransform)
    print(io, "FourierTransform{", eltype(ft), "}(")
    print(io, ft.modes, ", shift=", ft.shift, ")")
    return nothing
end

Base.ndims(T::FourierTransform) = length(T.modes)

function transform(ft::FourierTransform, x::AbstractArray)
    # res = rfft(x, 1:ndims(ft))
    res = rfft(real.(x), 1:ndims(ft))
    return res
end

function low_pass(ft::FourierTransform, x_fft::AbstractArray)
    return view(x_fft, (map(d -> 1:d, ft.modes)...), :, :)
end

truncate_modes(ft::FourierTransform, x_fft::AbstractArray) = low_pass(ft, x_fft)

function inverse(
    ft::FourierTransform, x_fft::AbstractArray{T,N}, x::AbstractArray{T2,N}
) where {T,T2,N}
    return real(irfft(x_fft, size(x, 1), 1:ndims(ft)))
end
