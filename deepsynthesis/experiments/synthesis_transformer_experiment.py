"""LTL Synthesis using the Transformer"""

import logging
import sys
import tensorflow as tf

from deepsynthesis import models
from deepsynthesis.data import ltl
from deepsynthesis.data.ast import TPEFormat
from deepsynthesis.data.expression import Notation
from deepsynthesis.experiments.synthesis_experiment import SynthesisExperiment
from deepsynthesis.optimization import lr_schedules

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynthesisTransformerExperiment(SynthesisExperiment):

    def __init__(self,
                 custom_pos_enc=True,
                 d_embed_enc=64,
                 d_embed_dec=None,
                 d_ff=128,
                 dropout=0.1,
                 ff_activation='relu',
                 num_heads=4,
                 num_layers_enc=6,
                 num_layers_dec=None,
                 warmup_steps=4000,
                 **kwargs):
        self.custom_pos_enc = custom_pos_enc
        if not custom_pos_enc:
            raise NotImplementedError
        self.d_embed_enc = d_embed_enc
        self.d_embed_dec = d_embed_dec if d_embed_dec else self.d_embed_enc
        self.d_ff = d_ff
        self.dropout = dropout
        self.ff_activation = ff_activation
        self.num_heads = num_heads
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec if num_layers_dec else self.num_layers_enc
        self.warmup_steps = warmup_steps
        if self.d_embed_enc % self.num_heads != 0:
            sys.exit(f'Encoder embedding dimension {self.d_embed_enc} is '
                     'not divisible by the number of attention heads'
                     f'{self.num_heads}')
        if self.d_embed_dec % self.num_heads != 0:
            sys.exit((f'Decoder embedding dimension {self.d_embed_dec} is '
                      'not divisible by the number of attention heads '
                      f'{self.num_heads}'))
        super().__init__(**kwargs)

    @property
    def abbr_name_args(self):
        result = super().abbr_name_args
        result.update({
            'custom_pos_enc': 'cpe',
            'd_ff': 'dff',
            'dropout': 'do',
            'ff_activation': '',
            'num_heads': 'nh',
            'warmup_steps': 'ws'
        })
        if self.d_embed_enc == self.d_embed_dec:
            result.update({'d_embed_enc': 'de'})
        else:
            result.update({'d_embed_enc': 'dee', 'd_embed_dec': 'ded'})
        if self.num_layers_enc == self.num_layers_dec:
            result.update({'num_layers_enc': 'nl'})
        else:
            result.update({'num_layers_enc': 'nle', 'num_layers_dec': 'nld'})
        return result

    @property
    def init_input_encoder(self):
        return ltl.specification.TreeEncoder(notation=Notation.INFIX,
                                             encoded_notation=Notation.PREFIX,
                                             start=False,
                                             eos=False,
                                             pad=self.max_input_length,
                                             tpe_format=TPEFormat.BRANCHDOWN,
                                             tpe_pad=self.d_embed_enc)

    @property
    def init_learning_rate(self):
        return lr_schedules.TransformerSchedule(self.d_embed_enc,
                                                warmup_steps=self.warmup_steps)

    @property
    def init_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                        beta_1=0.9,
                                        beta_2=0.98,
                                        epsilon=1e-9)

    def init_model(self, training=True):
        args = self.args
        args['input_vocab_size'] = self.input_vocab_size
        args['input_eos_id'] = self.input_eos_id
        args['input_pad_id'] = self.input_pad_id
        args['target_vocab_size'] = self.target_vocab_size
        args['target_start_id'] = self.target_start_id
        args['target_eos_id'] = self.target_eos_id
        args['target_pad_id'] = self.target_pad_id
        args['max_encode_length'] = self.max_input_length
        args['max_decode_length'] = self.max_target_length
        return models.transformer.create_model(
            args, training=training, custom_pos_enc=self.custom_pos_enc)

    def prepare_tf_dataset(self, tf_dataset):

        def shape_dataset(input_tensor, target_tensor):
            if self.custom_pos_enc:
                ltl_tensor, tpe_tensor = input_tensor
                return ((ltl_tensor, tpe_tensor, target_tensor), target_tensor)
            return ((input_tensor, target_tensor), target_tensor)

        return tf_dataset.map(shape_dataset)

    @classmethod
    def get_arg_parser(cls):
        parser = super().get_arg_parser()
        parser.description = 'Synthesis experiment using Transformer'
        defaults = cls.get_default_args()
        parser.add_argument('--d-embed-enc',
                            type=int,
                            default=defaults['d_embed_enc'])
        parser.add_argument('--d-embed-dec',
                            type=int,
                            default=defaults['d_embed_dec'])
        parser.add_argument('--d-ff', type=int, default=defaults['d_ff'])
        parser.add_argument('--dropout',
                            type=float,
                            default=defaults['dropout'])
        parser.add_argument('--ff-activation',
                            type=str,
                            default=defaults['ff_activation'])
        parser.add_argument('--num-heads',
                            type=int,
                            default=defaults['num_heads'])
        parser.add_argument('--num-layers-enc',
                            type=int,
                            default=defaults['num_layers_enc'])
        parser.add_argument('--num-layers-dec',
                            type=int,
                            default=defaults['num_layers_dec'])
        parser.add_argument('--warmup-steps',
                            type=int,
                            default=defaults['warmup_steps'])
        return parser


if __name__ == '__main__':
    SynthesisTransformerExperiment.cli()
