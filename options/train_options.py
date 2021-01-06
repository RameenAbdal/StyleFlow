from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.set_defaults(phase='train')
        parser.add_argument('--only_for_train', type=str, default='...')

        return parser