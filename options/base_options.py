import sys
import argparse



class BaseOptions():
    def __init__(self):
        pass

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='XXXX', help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='xxxx', help='which model to use')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--dataroot', type=str, default='./data/datasetX')


        self.initialized = True
        return parser



    def gather_options(self):
        # initialize parser with basic options

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt = parser.parse_args()
        self.parser = parser
        return opt




    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


    def parse(self):

        opt = self.gather_options()

        self.print_options(opt)



        self.opt = opt
        return self.opt
