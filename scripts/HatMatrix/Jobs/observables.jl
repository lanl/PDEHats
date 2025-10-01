##
function compute_bra_J_g_chi_g_J_ket()
    # Measurements
    bra_fns = [bra_C_smse, bra_C_mass, bra_C_energy]
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 10
    rtol = 5.0f-3
    idx_tuple_array = [(2, 2, 2), (3, 3, 3), (9, 9, 9)]
    ket_fn = ket_C_smse
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model, seed, N_ckpt, ket_fn, bra_fns, idx_tuple; rtol=rtol
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 10
    rtol = 5.0f-3
    idx_tuple_array = [(2, 2, 2), (3, 3, 3), (9, 9, 9)]
    ket_fn = ket_C_mass
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model, seed, N_ckpt, ket_fn, bra_fns, idx_tuple; rtol=rtol
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 35
    rtol = 5.0f-2
    idx_tuple_array = [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    ket_fn = ket_C_smse
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model, seed, N_ckpt, ket_fn, bra_fns, idx_tuple; rtol=rtol
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 35
    rtol = 5.0f-3
    idx_tuple_array = [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    ket_fn = ket_C_mass
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model, seed, N_ckpt, ket_fn, bra_fns, idx_tuple; rtol=rtol
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 42
    rtol = 5.0f-2
    idx_tuple_array = [(1, 1, 1), (2, 2, 2), (4, 4, 4)]
    ket_fn = ket_C_smse
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model, seed, N_ckpt, ket_fn, bra_fns, idx_tuple; rtol=rtol
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :ViT
    N_ckpt = 5
    seed = 42
    rtol = 5.0f-3
    idx_tuple_array = [(1, 1, 1), (4, 4, 4), (6, 6, 6)]
    ket_fn = ket_C_mass
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model, seed, N_ckpt, ket_fn, bra_fns, idx_tuple; rtol=rtol
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 10
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(2, 2, 2), (5, 5, 5), (9, 9, 9)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model, seed, N_ckpt, ket_fn, bra_fns, idx_tuple; rtol=rtol
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 10
    rtol = 5.0f-3
    ket_fn = ket_C_mass
    idx_tuple_array = [(2, 2, 2), (5, 5, 5), (9, 9, 9)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model, seed, N_ckpt, ket_fn, bra_fns, idx_tuple; rtol=rtol
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 35
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (3, 3, 3), (5, 5, 5)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model, seed, N_ckpt, ket_fn, bra_fns, idx_tuple; rtol=rtol
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 35
    rtol = 5.0f-3
    ket_fn = ket_C_mass
    idx_tuple_array = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model, seed, N_ckpt, ket_fn, bra_fns, idx_tuple; rtol=rtol
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 42
    rtol = 5.0f-3
    ket_fn = ket_C_smse
    idx_tuple_array = [(1, 1, 1), (5, 5, 5), (6, 6, 6)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model, seed, N_ckpt, ket_fn, bra_fns, idx_tuple; rtol=rtol
        )
    end
    ## Compute Ket/Batch/Model/Seed
    name_model = :UNet
    N_ckpt = 2
    seed = 42
    rtol = 5.0f-3
    ket_fn = ket_C_mass
    idx_tuple_array = [(1, 1, 1), (5, 5, 5), (6, 6, 6)]
    for idx_tuple in idx_tuple_array
        compute_bra_J_g_chi_g_J_ket(
            name_model, seed, N_ckpt, ket_fn, bra_fns, idx_tuple; rtol=rtol
        )
    end
    ##
    return nothing
end
