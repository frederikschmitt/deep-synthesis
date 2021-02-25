"""Base experiment class"""

import argparse
import inspect
import json
import logging
import os
import sys

import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dtype_float_str_to_class = {
    'float16': tf.float16,
    'float32': tf.float32,
    'float64': tf.float64
}
dtype_float_class_to_str = {c: s for s, c in dtype_float_str_to_class.items()}

dtype_int_str_to_class = {
    'int16': tf.int16,
    'int32': tf.int32,
    'int64': tf.int64
}
dtype_int_class_to_str = {c: s for s, c in dtype_int_str_to_class.items()}


class Experiment():

    def __init__(self,
                 batch_size=32,
                 cache_dataset=True,
                 checkpoint_monitor='val_loss',
                 dataset='/deepsynthesis/data/SC100',
                 dtype_float='float32',
                 dtype_int='int32',
                 initial_epoch=0,
                 model_dir='/deepsynthesis/models',
                 name=None,
                 shuffle_on_load=False,
                 tf_shuffle_buffer_size=0,
                 user_dir='~',
                 validation_freq=1):
        self.batch_size = batch_size
        self.cache_dataset = cache_dataset
        self.checkpoint_monitor = checkpoint_monitor
        self.data_dir = dataset
        if dtype_float not in dtype_float_str_to_class:
            sys.exit(f'Unrecognized float data type argument {dtype_float}')
        self.dtype_float = dtype_float_str_to_class[dtype_float]
        if dtype_int not in dtype_int_str_to_class:
            sys.exit(f'Unrecognized integer data type argument {dtype_int}')
        self.dtype_int = dtype_int_str_to_class[dtype_int]
        self.initial_epoch = initial_epoch
        self.name = name if name else self.computed_name
        self.model_dir = os.path.join(model_dir, self.name)
        self.shuffle_on_load = shuffle_on_load
        self.tf_shuffle_buffer_size = tf_shuffle_buffer_size
        self.user_dir = os.path.expanduser(user_dir)
        self.validation_freq = validation_freq

        self._dataset = None
        self._eval_model = None
        self._learning_rate = None
        self._optimizer = None
        self._prepared_tf_dataset = {}
        self._tf_dataset = {}
        self._train_model = None
        self._verifier = None

        logger.info('Initialized experiment with arguments:\n%s',
                    '\n'.join([f'{a}: {v}' for a, v in self.args.items()]))

        if os.path.exists(self.model_dir):
            logger.info('Found exisiting model directory %s', self.model_dir)
        else:
            #create model directory
            os.makedirs(self.model_dir)
            logger.info('Created model directory %s', self.model_dir)

        if os.path.exists(self.args_filepath):
            #Checking args file:
            mismatches = 0
            with open(self.args_filepath) as args_file:
                loaded_args = json.load(args_file)
                for arg, value in self.str_args.items():
                    if arg not in loaded_args:
                        mismatches += 1
                        logger.warning(
                            'Argument %s with value %s could not be found in argument file',
                            arg, value)
                    elif loaded_args[arg] != value:
                        mismatches += 1
                        logger.warning(
                            'Argument %s with value %s does not match argument file value %s',
                            arg, value, loaded_args[arg])
            logger.info('Mismatches with arguments file: %d', mismatches)
        else:
            self.save_args()

    @property
    def args(self):
        return {
            k: v
            for k, v in sorted(self.__dict__.items())
            if not k.startswith('_')
        }

    @property
    def args_filepath(self):
        return os.path.join(self.model_dir, 'args.json')

    @property
    def abbr_name_args(self):
        return {'batch_size': 'bs'}

    @property
    def callbacks(self):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            monitor=self.checkpoint_monitor,
            save_weights_only=True,
            save_best_only=True,
            verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.model_dir)
        return [checkpoint_callback, tensorboard_callback]

    @property
    def checkpoint_filepath(self):
        return os.path.join(self.model_dir, 'checkpoint')

    @property
    def computed_name(self):
        default_args = self.get_default_args()
        abbr_args_values = []
        for arg, abbr in self.abbr_name_args.items():
            value = getattr(self, arg)
            if default_args[arg] == value:
                continue
            arg_abbr_value = arg + '_abbr_value'
            if hasattr(self, arg_abbr_value):
                value = getattr(self, arg_abbr_value)
            if isinstance(value, bool):
                abbr_args_values.append(abbr if value else f'n{abbr}')
            else:
                abbr_args_values.append((f'{abbr}{value}'))
        if not abbr_args_values:
            return 'default'
        abbr_args_values.sort()
        return '-'.join(abbr_args_values)

    @property
    def dataset(self):
        if self._dataset:
            return self._dataset
        self._dataset = self.init_dataset
        logger.info('Constructed dataset from directory %s', self.data_dir)
        if self.shuffle_on_load:
            self._dataset.shuffle()
            logger.info('Shuffled dataset')
        return self._dataset

    def eval(self, split, steps=None, training=False, verify=False):
        generator = self.dataset.generator(splits=[split])
        name = f'eval-{split}-n{steps * self.batch_size}'
        self.eval_generator(generator,
                            name,
                            includes_target=True,
                            steps=steps,
                            training=training,
                            verify=verify)

    def eval_dataset(self, dataset, steps=None, training=False, verify=False):
        raise NotImplementedError

    def eval_generator(self,
                       generator,
                       name,
                       includes_target=False,
                       steps=None,
                       training=False,
                       verify=False):
        raise NotImplementedError

    @property
    def eval_model(self):
        if not self._eval_model:
            self._eval_model = self.init_model(training=False)
            logger.info('Created evaluation model')
            checkpoint = tf.train.latest_checkpoint(self.model_dir)
            if checkpoint:
                logger.info('Found checkpoint %s', checkpoint)
                self._eval_model.load_weights(checkpoint).expect_partial()
                logger.info('Loaded weights from checkpoint')
        return self._eval_model

    @property
    def init_dataset(self):
        raise NotImplementedError

    @property
    def init_learning_rate(self):
        raise NotImplementedError

    def init_model(self, training=True):
        raise NotImplementedError

    @property
    def init_optimizer(self):
        raise NotImplementedError

    @property
    def init_tf_dataset(self):
        raise NotImplementedError

    @property
    def init_verifier(self):
        raise NotImplementedError

    @property
    def learning_rate(self):
        if not self._learning_rate:
            self._learning_rate = self.init_learning_rate
        return self._learning_rate

    @property
    def optimizer(self):
        if not self._optimizer:
            self._optimizer = self.init_optimizer
        return self._optimizer

    def prepare_tf_dataset(self, tf_dataset):
        return tf_dataset

    def prepared_tf_dataset(self, split):
        if not split in self._prepared_tf_dataset:
            dataset = self.prepare_tf_dataset(self.tf_dataset(split))
            if self.cache_dataset:
                dataset = dataset.cache()
            if self.tf_shuffle_buffer_size:
                dataset = dataset.shuffle(self.tf_shuffle_buffer_size,
                                          reshuffle_each_iteration=False)
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            dataset = dataset.prefetch(2)
            self._prepared_tf_dataset[split] = dataset
        return self._prepared_tf_dataset[split]

    def run(self, epochs=1):
        history = self.train_model.fit(
            self.prepared_tf_dataset('train'),
            callbacks=self.callbacks,
            epochs=self.initial_epoch + epochs,
            initial_epoch=self.initial_epoch,
            validation_data=self.prepared_tf_dataset('val'),
            validation_freq=self.validation_freq)
        history_filepath = os.path.join(self.model_dir, 'history.json')
        with open(history_filepath, 'w') as history_file:
            json.dump(history.history, history_file, indent=2)
            logger.info('Written training history to %s', history_filepath)
        self.initial_epoch += epochs
        self.save_args()
        self._eval_model = None

    @property
    def storage_dir(self):
        return os.path.join(self.user_dir, 'deeplogic-storage')

    @property
    def str_args(self):
        result = self.args
        result['dtype_float'] = dtype_float_class_to_str[self.dtype_float]
        result['dtype_int'] = dtype_int_class_to_str[self.dtype_int]
        return result

    def save_args(self, filepath=None):
        if not filepath:
            filepath = self.args_filepath
        with open(filepath, 'w') as args_file:
            json.dump(self.str_args, args_file, indent=2)
        logger.info('Written arguments to %s', filepath)

    @property
    def temp_dir(self):
        temp_path = os.path.join(self.storage_dir, 'temp')
        if not os.path.isdir(temp_path):
            os.makedirs(temp_path)
        return temp_path

    def tf_dataset(self, split):
        if not self._tf_dataset:
            self._tf_dataset = self.init_tf_dataset
        return self._tf_dataset[split]

    @property
    def train_model(self):
        if not self._train_model:
            self._train_model = self.init_model(training=True)
            logger.info('Created training model')
            checkpoint = tf.train.latest_checkpoint(self.model_dir)
            if checkpoint:
                logger.info('Found checkpoint %s', checkpoint)
                self._train_model.load_weights(checkpoint).expect_partial()
                logger.info('Loaded weights from checkpoint')
            self._train_model.compile(optimizer=self.optimizer)
            logger.info('Compiled training model')
        return self._train_model

    @property
    def verifier(self):
        if not self._verifier:
            self._verifier = self.init_verifier
        return self._verifier

    @classmethod
    def get_arg_parser(cls):
        parser = argparse.ArgumentParser(description='DeepLogic experiment')
        defaults = cls.get_default_args()
        parser.add_argument('--batch-size',
                            type=int,
                            default=defaults['batch_size'])
        parser.add_argument('--no-dataset-cache',
                            action='store_false',
                            dest='cache_dataset')
        parser.add_argument('--checkpoint-monitor',
                            type=str,
                            default=defaults['checkpoint_monitor'])
        parser.add_argument('-d',
                            '--dataset',
                            default=defaults['dataset'])
        parser.add_argument('--dtype-float',
                            default='float32',
                            choices=dtype_int_str_to_class.keys())
        parser.add_argument('--dtype-int',
                            default='int32',
                            choices=dtype_int_str_to_class.keys())
        parser.add_argument('--initial-epoch',
                            type=int,
                            default=defaults['initial_epoch'])
        parser.add_argument('-n', '--name', default=defaults['name'])
        parser.add_argument('--shuffle-on-load', action='store_true')
        parser.add_argument('--tf-shuffle-buffer-size',
                            type=int,
                            default=defaults['tf_shuffle_buffer_size'])
        parser.add_argument('--user-dir', default=defaults['user_dir'])
        return parser

    @classmethod
    def cli(cls):
        parser = cls.get_arg_parser()
        parser.add_argument('--epochs', type=int, default=1)
        args = parser.parse_args()
        init_args = vars(args)
        epochs = init_args.pop('epochs')
        experiment = cls(**init_args)
        experiment.run(epochs)

    @classmethod
    def get_default_args(cls):
        default_args = {}
        for super_class in reversed(cls.mro()):
            signature = inspect.signature(super_class.__init__)
            default_args.update(
                {k: v.default for k, v in signature.parameters.items()})
        for special_arg in ['self', 'args', 'kwargs']:
            default_args.pop(special_arg)
        return default_args

    @classmethod
    def load_from_dir(cls, model_dir):
        args_filepath = os.path.join(model_dir, 'args.json')
        if os.path.isfile(args_filepath):
            with open(args_filepath, 'r') as args_file:
                args = json.load(args_file)
                experiment = cls(**args)
                return experiment
        logger.critical('Could not locate arguments file %s', args_filepath)
        return None
