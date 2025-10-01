##
function _println(should_log::Bool, msg::String)
    return should_log && println("[$(now())] ", msg)
end
function _print(should_log::Bool, msg::String)
    return should_log && print("[$(now())] ", msg)
end
##
function _sample_replicate(_rng::AbstractRNG, n::Int)
    rng = Lux.replicate(_rng)
    for i in 1:n
        rand(rng)
    end
    return Lux.replicate(rng)
end
##
function _nameof(s::String)
    replacements = Dict(
        "loss_mse_scaled" => "Mean Squared Error (Scaled)",
        "loss_ssim_scaled" => "SSIM (Scaled)",
    )
    return haskey(replacements, s) ? replacements[s] : s
end
