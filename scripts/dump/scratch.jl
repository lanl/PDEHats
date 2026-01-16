##
name_model = :ViT
seed = 10
N_ckpt = 5
idx_tuple_array = [(2, 2, 2), (3, 3, 3), (9, 9, 9)]
loss_fn = PDEHats.loss_mse_scaled
ket_fn = ket_C_mass
ket_g = g_identity
lambda = 1.0f-6
rtol = 5.0f-3
T_max = 17
dev = gpu_device()
using NPZ
_data_RP = PDEHats._get_data("RP")[:, :, :, :, 1:25];
_data_RPUI = PDEHats._get_data("RPUI")[:, :, :, :, 1:25];
_data_CRP = PDEHats._get_data("CRP")[:, :, :, :, 1:25];
readme = codeunits(
    "This array is expected to have shape (100, 21, 5, 128, 128). The index of dimension 5 represents density, velocity-x, velocity-y, pressure, energy, respectively. Care should be taken to match which spatial index is the x-index, if necessary.",
)
##
data_dict = Dict(
    "RP" => _data_RP,
    "RPUI" => _data_RPUI,
    "CRP" => _data_CRP,
    "README" => readme,
)
##
npzwrite(projectdir("data_RP_RPUI_CRP.npz"), data_dict)
##
prefix = "bra_C_mass"
suffix = ".jld2"
dir = projectdir("results/HatMatrix/NS")
paths = PDEHats.find_files(dir, prefix, suffix)
rm.(paths)
##
prefix = "bra_C_energy"
suffix = ".jld2"
dir = projectdir("results/HatMatrix/NS")
paths = PDEHats.find_files(dir, prefix, suffix)
rm.(paths)
##
suffix = "eigen_min.jld2"
dir = projectdir("results/Eigen/")
paths = PDEHats.find_files_by_suffix(dir, suffix)
rm.(paths)
##
p = load(
    readdir(
        projectdir(
            "results/HatMatrix/NS/batches_test/seed_10_ratiotrain_65f-2_ratioval_5f-2/Tmax_17/",
        );
        join=true,
    )[1],
)
input = p["input"];
target = p["target"];
##
name_model = :UNet
name_data = :NS
g_identity = identity
ket_g = g_identity
ket_fn = ket_C_smse
seed = 10
idx_NT = (; idx_rp=1, idx_crp=1, idx_rpui=1)
epoch = 100
ratio_train = 0.65f0
ratio_val = 0.05f0
lambda = 1.0f-4
T_max = 17
loss_fn = loss_smse
##
m_x, _ = model(input[:, :, :, 1:1, 1:1], ps, st);
mean(abs2.(m_x .- target))
PDEHats.loss_mse_scaled(target, m_x)
loss_smse(input, target, m_x)
##
input_r = g_rotate_90(input);
m_x_r, _ = model(input_r[:, :, :, 1:1, 3:3], ps, st);
target_r = g_rotate_90(target);
mean(abs2.(m_x_r .- target_r))
PDEHats.loss_mse_scaled(target_r, m_x_r)
loss_smse(input_r, target_r, m_x_r)
##
obj_fn = PDEHats.loss_mse_scaled
obs = (input, target)
opt = AdamW(; eta=1.0f0, lambda=1.0f-4)
state_train = Training.TrainState(model, ps, st, opt)
grads, loss, _, state_train = Training.compute_gradients(
    AutoZygote(), obj_fn, obs, state_train
)
##
suffix = "eigen_min.jld2"
dir = projectdir("results/Eigen/")
paths = PDEHats.find_files_by_suffix(dir, suffix)
# rm.(paths)
##
t = 9
b = 2
f = load(
    readdir(
        "results/HatMatrix/NS/batches_test/seed_10_ratiotrain_65f-2_ratioval_5f-2/Tmax_17";
        join=true,
    )[1],
)["input"][
    :, :, 1, t, b
];
heatmap(f)
