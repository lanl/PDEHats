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
