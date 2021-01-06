from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.set_defaults(phase='test')
        parser.add_argument('--only_for_test', type=str, default='...')
        parser.add_argument('--network_pkl', type=str, default='gdrive:networks/stylegan2-ffhq-config-f.pkl')
        parser.add_argument('--max_result_snapshots', default=30, help='max result snapshots')
        return parser