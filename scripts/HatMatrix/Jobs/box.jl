##
function get_box(; L::Int=16, l::Int=8)
    box_full = collect(Iterators.product((-L):L, (-L):L))
    box_done = collect(Iterators.product((-l):l, (-l):l))
    box_todo = setdiff(box_full, box_done)
    translations = map(box_todo) do (dx, dy)
        if (dx == 0) && (dy == 0)
            fname = Symbol("g_identity")
        else
            fname = Symbol("g_shift_x_$(dx)_y_$(dy)")
        end
        @eval return $(fname)
    end
    return vec(translations)
end
##
function compute_box_U()
    # Measurements
    bra_fns = [bra_C_smse]
    bra_gs = [get_box()...]
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 10
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(9, 9, 9)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model,
            seed,
            N_ckpt,
            ket_fn,
            bra_fns,
            idx_tuple;
            rtol=rtol,
            bra_gs=bra_gs,
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 35
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(5, 5, 5)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model,
            seed,
            N_ckpt,
            ket_fn,
            bra_fns,
            idx_tuple;
            rtol=rtol,
            bra_gs=bra_gs,
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 42
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(6, 6, 6)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model,
            seed,
            N_ckpt,
            ket_fn,
            bra_fns,
            idx_tuple;
            rtol=rtol,
            bra_gs=bra_gs,
        )
    end
    return nothing
end
function compute_box_U10()
    # Measurements
    bra_fns = [bra_C_smse]
    bra_gs = [get_box()...]
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 10
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(2, 2, 2), (5, 5, 5), (9, 9, 9)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model,
            seed,
            N_ckpt,
            ket_fn,
            bra_fns,
            idx_tuple;
            rtol=rtol,
            bra_gs=bra_gs,
        )
    end
    return nothing
end
function compute_box_U35()
    # Measurements
    bra_fns = [bra_C_smse]
    bra_gs = [get_box()...]
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 35
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model,
            seed,
            N_ckpt,
            ket_fn,
            bra_fns,
            idx_tuple;
            rtol=rtol,
            bra_gs=bra_gs,
        )
    end
    return nothing
end
function compute_box_U42()
    # Measurements
    bra_fns = [bra_C_smse]
    bra_gs = [get_box()...]
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 42
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (5, 5, 5), (6, 6, 6)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model,
            seed,
            N_ckpt,
            ket_fn,
            bra_fns,
            idx_tuple;
            rtol=rtol,
            bra_gs=bra_gs,
        )
    end
    return nothing
end
function compute_box_V10()
    # Measurements
    bra_fns = [bra_C_smse]
    bra_gs = [get_box()...]
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 10
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(2, 2, 2), (3, 3, 3), (9, 9, 9)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model,
            seed,
            N_ckpt,
            ket_fn,
            bra_fns,
            idx_tuple;
            rtol=rtol,
            bra_gs=bra_gs,
        )
    end
    return nothing
end
function compute_box_V35()
    # Measurements
    bra_fns = [bra_C_smse]
    bra_gs = [get_box()...]
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 35
    rtol = 5.0f-2
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model,
            seed,
            N_ckpt,
            ket_fn,
            bra_fns,
            idx_tuple;
            rtol=rtol,
            bra_gs=bra_gs,
        )
    end
    return nothing
end
function compute_box_V42()
    # measurements
    bra_fns = [bra_C_smse]
    bra_gs = [get_box()...]
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 42
    rtol = 5.0f-2
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (2, 2, 2)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model,
            seed,
            N_ckpt,
            ket_fn,
            bra_fns,
            idx_tuple;
            rtol=rtol,
            bra_gs=bra_gs,
        )
    end
    return nothing
end
