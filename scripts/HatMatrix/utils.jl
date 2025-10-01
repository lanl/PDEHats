##
function string_rtol(rtol::AbstractFloat)
    _rtol = @sprintf "%.0e" rtol
    __rtol = replace(_rtol, "-0" => "-")
    return __rtol
end
