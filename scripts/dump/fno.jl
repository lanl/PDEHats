##
using DrWatson
@quickactivate :PDEHats
##
using Lux
using AbstractFFTs
using Random
using ConcreteStructs
using Optimisers
using Statistics
using ComponentArrays
##
rng = Xoshiro()
modes = (32, 32)
in_channels = 4
out_channels = 4
hidden_channels = 2
x = randn(Float32, 128, 128, in_channels, 3);
#
m = PDEHats.FNO(modes, in_channels, out_channels, hidden_channels)
m = PDEHats.FourierNeuralOperator(
    modes, in_channels, out_channels, hidden_channels; use_channel_mlp=false
)
ps, st = Lux.setup(rng, m)
ps = ComponentArray(ps)
m_x, _ = m(x, ps, st);
size(m_x);
y = randn(Float32, size(m_x));
##
opt = AdamW()
backend_autodiff = AutoZygote()
obs = (x, y)
obj_fn = MSELoss()
state_train = Training.TrainState(m, ps, st, opt)
grads, loss, _, state_train = Training.compute_gradients(
    backend_autodiff, obj_fn, obs, state_train
)
grads_rms = sqrt(mean(abs2.(getdata(ComponentArray(grads)))))
println("[$(nameof(obj_fn))] RMS Grads: $(grads_rms)")
@assert grads_rms > 0.0f0
##
stabilizer = tanh
act = gelu
# act = identity
ch = hidden_channels => hidden_channels
in_chs, out_chs = ch
shift = false
fno_skip = :linear
_transform = PDEHats.FourierTransform{ComplexF32}(modes, shift)
stabilizer = WrappedFunction(Base.BroadcastFunction(stabilizer))
conv_layer = PDEHats.OperatorConv(ch, modes, _transform)
fno_skip_layer = PDEHats.__fno_skip_connection(in_chs, out_chs, false, fno_skip)
m = PDEHats.OperatorKernel(
    Parallel(
        PDEHats.Fix1(PDEHats.add_act, act),
        fno_skip_layer,
        Chain(; stabilizer, conv_layer),
    ),
)
ps, st = Lux.setup(rng, m)
ps = ComponentArray(ps)
#
x = randn(Float32, 128, 128, hidden_channels, 3);
m_x, _ = m(x, ps, st);
size(m_x);
y = randn(Float32, size(m_x));
##
opt = AdamW()
backend_autodiff = AutoZygote()
obs = (x, y)
obj_fn = MSELoss()
state_train = Training.TrainState(m, ps, st, opt)
grads, loss, _, state_train = Training.compute_gradients(
    backend_autodiff, obj_fn, obs, state_train
)
grads_rms = sqrt(mean(abs2.(getdata(ComponentArray(grads)))))
println("[$(nameof(obj_fn))] RMS Grads: $(grads_rms)")
@assert grads_rms > 0.0f0
