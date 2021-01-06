from .odefunc import ODEfunc, ODEnet
from .normalization import MovingBatchNorm1d
from .cnf import CNF, SequentialFlow


def count_nfe(model):
    class AccNumEvals(object):

        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, CNF):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_time(model):
    class Accumulator(object):

        def __init__(self):
            self.total_time = 0

        def __call__(self, module):
            if isinstance(module, CNF):
                self.total_time = self.total_time + module.sqrt_end_time * module.sqrt_end_time

    accumulator = Accumulator()
    model.apply(accumulator)
    return accumulator.total_time


def build_model( input_dim, hidden_dims, context_dim, num_blocks, conditional):
    def build_cnf():
        diffeq = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(input_dim,),
            context_dim=context_dim,
            layer_type='concatsquash',
            nonlinearity='tanh',
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=1.0,
            train_T=True,
            conditional=conditional,
            solver='dopri5',
            use_adjoint=True,
            atol=1e-5,
            rtol=1e-5,
        )
        return cnf

    chain = [build_cnf() for _ in range(num_blocks)]
    bn_layers = [MovingBatchNorm1d(input_dim, bn_lag=0, sync=False)
                     for _ in range(num_blocks)]
    bn_chain = [MovingBatchNorm1d(input_dim, bn_lag=0, sync=False)]
    for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
    chain = bn_chain
    model = SequentialFlow(chain)

    return model


def cnf(input_dim,dims,zdim,num_blocks):
    dims = tuple(map(int, dims.split("-")))
    model = build_model(input_dim, dims, zdim, num_blocks, True).cuda()
    print("Number of trainable parameters of Point CNF: {}".format(count_parameters(model)))
    return model


