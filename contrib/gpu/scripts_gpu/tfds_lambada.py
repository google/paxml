# coding=utf-8
# Copyright 2022 The Pax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adapted from: https://github.com/EleutherAI/the-pile
lambada dataset
"""

import os

import tensorflow as tf
import tensorflow_datasets as tfds

import io
import zstandard
import jsonlines
import simdjson as json
from itertools import chain

## to load data:
# import tensorflow as tf
# import tensorflow_datasets as tfds
# from paxml.paxml.tasks.lm.params import tfds_lambada

# dataset = tfds.load('MyLambada',  data_dir='lambada_pile_preproc')


_DESCRIPTION = """The LAMBADA dataset evaluates the capabilities of computational
models for text understanding by means of a word prediction task. LAMBADA is a
collection of narrative passages sharing the characteristic that human subjects
are able to guess their last word if they are exposed to the whole passage, but
not if they only see the last sentence preceding the target word
"""

_CITATION = """@inproceedings{paperno-etal-2016-lambada,
    title = "The {LAMBADA} dataset: Word prediction requiring a broad discourse context",
    author = "Paperno, Denis  and
      Kruszewski, Germ{\'a}n  and
      Lazaridou, Angeliki  and
      Pham, Ngoc Quan  and
      Bernardi, Raffaella  and
      Pezzelle, Sandro  and
      Baroni, Marco  and
      Boleda, Gemma  and
      Fern{\'a}ndez, Raquel",
    booktitle = "Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2016",
    address = "Berlin, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P16-1144",
    doi = "10.18653/v1/P16-1144",
    pages = "1525--1534",
}
"""

_LAMBADA_DATASET_URL = 'https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl'

try:
    import simdjson as json
except ImportError:
    print('Installing simdjson library')
    os.system('pip install -q pysimdjson')
    import simdjson as json
    parser = json.Parser()


_CITATION = """
"""
_DATASET_MODES = ["lm"]

_URLS = {
    'my_lambada': {
        'test': _LAMBADA_DATASET_URL,
    }
}


_VERSION = tfds.core.Version('1.0.0')
_RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
}

_NAME = 'my_lambada'
_FILE_FORMAT = 'jsonlines'

def json_parser(x):
    try:
        line = parser.parse(x).as_dict()
        return line
    except ValueError:
        return x

class LambadaReader:
    def __init__(self, filenames, para_joiner='\n\n'):
        if not isinstance(filenames, list):
            filenames = [filenames]
        self.filenames = filenames
        self.para_joiner = para_joiner

    def _read_fn(self, filename):
        with tf.io.gfile.GFile(filename, 'rb+') as f:
            reader = jsonlines.Reader(f)
            for item in reader:
                result = dict()
                text = item['text']
                text = text.replace("“", '"')
                text = text.replace("”", '"')
                text = text.replace("’", "'")
                text = text.replace("‘", "'")
                last_token = text.split()[-1]
                start_idx = text.rfind(last_token)
                beginning_tokens = text[:start_idx].strip()
                last_token = ' ' + last_token.strip()
                
                result['text'] = beginning_tokens
                result['target'] = last_token

                yield result
    
    def __iter__(self):
        for filename in self.filenames:
            return self._read_fn(filename)


class LambadaConfig(tfds.core.BuilderConfig):
    def __init__(self, *, mode=None, **kwargs):
        super(LambadaConfig, self).__init__(
            name=mode,
            description="Lambada dataset",
            **kwargs)

class MyLambada(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        LambadaConfig(version=_VERSION, mode=mode) for mode in _DATASET_MODES
    ]
    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'inputs': tfds.features.Text(),
                'targets': tfds.features.Text()
            }),
            supervised_keys=None,
            homepage='https://zenodo.org/record/2630551#.X4Xzn5NKjUI',
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        dl_manager.verify_ssl = False
        dl_paths = dl_manager.download(_URLS['my_lambada'])
        return {
            'test': self._generate_examples(dl_paths['test'])
        }

    def _generate_examples(self, paths):
        pipeline = LambadaReader(paths)
        for x, result in enumerate(pipeline):
            if result:
                idx = f'{x}_my_lambada'
                yield idx, {'inputs': result['text'], 'targets': result['target']}

