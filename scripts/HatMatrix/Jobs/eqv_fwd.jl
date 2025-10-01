##
function compute_eqv_fwd_ViT()
    ## Compute Model/Seed/Batch
    name_model = :ViT
    N_ckpt = 5
    rtol = 5.0f-3
    seed = 10
    ket_fn = ket_C_smse
    idx_tuple_array = [(2, 2, 2), (3, 3, 3), (9, 9, 9)]
    for idx_tuple in idx_tuple_array
        compute_eqv_fwd(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Model/Seed/Batch
    name_model = :ViT
    N_ckpt = 5
    rtol = 5.0f-3
    seed = 10
    ket_fn = ket_C_mass
    idx_tuple_array = [(2, 2, 2), (3, 3, 3), (9, 9, 9)]
    for idx_tuple in idx_tuple_array
        compute_eqv_fwd(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Model/Seed/Batch
    name_model = :ViT
    N_ckpt = 5
    rtol = 5.0f-2
    seed = 35
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    for idx_tuple in idx_tuple_array
        compute_eqv_fwd(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Model/Seed/Batch
    name_model = :ViT
    N_ckpt = 5
    rtol = 5.0f-3
    seed = 35
    ket_fn = ket_C_mass
    idx_tuple_array = [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    for idx_tuple in idx_tuple_array
        compute_eqv_fwd(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Model/Seed/Batch
    name_model = :ViT
    N_ckpt = 5
    rtol = 5.0f-2
    seed = 42
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (2, 2, 2), (4, 4, 4)]
    for idx_tuple in idx_tuple_array
        compute_eqv_fwd(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Model/Seed/Batch
    name_model = :ViT
    N_ckpt = 5
    rtol = 5.0f-3
    seed = 42
    ket_fn = ket_C_mass
    idx_tuple_array = [(1, 1, 1), (4, 4, 4), (6, 6, 6)]
    for idx_tuple in idx_tuple_array
        compute_eqv_fwd(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    return nothing
end
function compute_eqv_fwd_UNet()
    ## Compute Model/Seed/Batch
    name_model = :UNet
    N_ckpt = 2
    rtol = 5.0f-3
    seed = 10
    ket_fn = ket_C_smse
    idx_tuple_array = [(2, 2, 2), (5, 5, 5), (9, 9, 9)]
    for idx_tuple in idx_tuple_array
        compute_eqv_fwd(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Model/Seed/Batch
    name_model = :UNet
    N_ckpt = 2
    rtol = 5.0f-3
    seed = 10
    ket_fn = ket_C_mass
    idx_tuple_array = [(2, 2, 2), (5, 5, 5), (9, 9, 9)]
    for idx_tuple in idx_tuple_array
        compute_eqv_fwd(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Model/Seed/Batch
    name_model = :UNet
    N_ckpt = 2
    rtol = 5.0f-3
    seed = 35
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    for idx_tuple in idx_tuple_array
        compute_eqv_fwd(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Model/Seed/Batch
    name_model = :UNet
    N_ckpt = 2
    rtol = 5.0f-3
    seed = 35
    ket_fn = ket_C_mass
    idx_tuple_array = [(1, 1, 1), (3, 3, 3), (2, 2, 2)]
    for idx_tuple in idx_tuple_array
        compute_eqv_fwd(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Model/Seed/Batch
    name_model = :UNet
    N_ckpt = 2
    rtol = 5.0f-3
    seed = 42
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (5, 5, 5), (6, 6, 6)]
    for idx_tuple in idx_tuple_array
        compute_eqv_fwd(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ## Compute Model/Seed/Batch
    name_model = :UNet
    N_ckpt = 2
    rtol = 5.0f-3
    seed = 42
    ket_fn = ket_C_mass
    idx_tuple_array = [(1, 1, 1), (5, 5, 5), (6, 6, 6)]
    for idx_tuple in idx_tuple_array
        compute_eqv_fwd(name_model, seed, N_ckpt, ket_fn, idx_tuple; rtol=rtol)
    end
    ##
    return nothing
end
