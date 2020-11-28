# from multiprocessing.pool import Pool
from enum import Enum
import os
import data.bert.modeling as modeling
import json
import contextlib
import tempfile
import re
import collections
import unicodedata
import six
import tensorflow as tf
import numpy as np


BERT_DIR = '/Users/yuanxiaoming/model/uncased_L-4_H-128_A-2/'


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        # self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_lst_to_features(lst_str, max_seq_length, max_position_embeddings,
                            tokenizer, is_tokenized=False, mask_cls_sep=False):
    """Loads a data file into a list of `InputBatch`s."""

    examples = read_tokenized_examples(lst_str) if is_tokenized else read_examples(lst_str)

    _tokenize = lambda x: tokenizer.mark_unk_tokens(x) if is_tokenized else tokenizer.tokenize(x)

    all_tokens = [(_tokenize(ex.text_a), _tokenize(ex.text_b) if ex.text_b else []) for ex in examples]

    # user did not specify a meaningful sequence length
    # override the sequence length by the maximum seq length of the current batch
    if max_seq_length is None:
        max_seq_length = max(len(ta) + len(tb) for ta, tb in all_tokens)
        # add special tokens into account
        # case 1: Account for [CLS], tokens_a [SEP], tokens_b [SEP] -> 3 additional tokens
        # case 2: Account for [CLS], tokens_a [SEP] -> 2 additional tokens
        max_seq_length += 3 if any(len(tb) for _, tb in all_tokens) else 2
        max_seq_length = min(max_seq_length, max_position_embeddings)
        print('"max_seq_length" is undefined, and bert config json defines "max_position_embeddings"=%d. '
              'hence set "max_seq_length"=%d according to the current batch.' % (
                  max_position_embeddings, max_seq_length))

    for (tokens_a, tokens_b) in all_tokens:
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        input_type_ids = [0] * len(tokens)
        input_mask = [int(not mask_cls_sep)] + [1] * len(tokens_a) + [int(not mask_cls_sep)]

        if tokens_b:
            tokens += tokens_b + ['[SEP]']
            input_type_ids += [1] * (len(tokens_b) + 1)
            input_mask += [1] * len(tokens_b) + [int(not mask_cls_sep)]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length. more pythonic
        pad_len = max_seq_length - len(input_ids)
        input_ids += [0] * pad_len
        input_mask += [0] * pad_len
        input_type_ids += [0] * pad_len

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(input_type_ids) == max_seq_length

        # print('tokens: %s' % ' '.join([printable_text(x) for x in tokens]))
        # print('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
        # print('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
        # print('input_type_ids: %s' % ' '.join([str(x) for x in input_type_ids]))

        yield InputFeatures(
            # unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(lst_strs):
    """Read a list of `InputExample`s from a list of strings."""
    unique_id = 0
    for ss in lst_strs:
        line = convert_to_unicode(ss)
        if not line:
            continue
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1


def read_tokenized_examples(lst_strs):
    unique_id = 0
    lst_strs = [[convert_to_unicode(w) for w in s] for s in lst_strs]
    for ss in lst_strs:
        text_a = ss
        text_b = None
        try:
            j = ss.index('|||')
            text_a = ss[:j]
            text_b = ss[(j + 1):]
        except ValueError:
            pass
        yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, str):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, str):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    def mark_unk_tokens(self, tokens, unk_token='[UNK]'):
        return [t if t in self.vocab else unk_token for t in tokens]


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class Arguments:
    def __init__(self):
        self.max_seq_len = 50
        self.verbose = False
        self.model_dir = BERT_DIR
        self.config_name = 'bert_config.json'
        self.tuned_model_dir = None
        self.ckpt_name = 'bert_model.ckpt'
        self.fp16 = False
        self.xla = False
        self.pooling_layer = [-2]
        self.pooling_strategy = PoolingStrategy.REDUCE_MEAN
        self.graph_tmp_dir = None
        self.num_worker = 1
        self.max_batch_size = 256
        self.gpu_memory_fraction = 0.5
        self.mask_cls_sep = False
        self.show_tokens_to_client = False


class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    FIRST_TOKEN = 4  # corresponds to [CLS] for single sequences
    LAST_TOKEN = 5  # corresponds to [SEP] for single sequences
    CLS_TOKEN = 4  # corresponds to the first token for single seq.
    SEP_TOKEN = 5  # corresponds to the last token for single seq.

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()


class BertWorker:
    def __init__(self, args, graph_path, graph_config):
        # super().__init__()
        self.device_id = -1
        self.max_seq_len = args.max_seq_len
        self.mask_cls_sep = args.mask_cls_sep
        self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.model_dir = args.model_dir
        self.verbose = args.verbose
        self.graph_path = graph_path
        self.bert_config = graph_config
        self.use_fp16 = args.fp16
        self.show_tokens_to_client = args.show_tokens_to_client

    def get_estimator(self, tf):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec

        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            input_names = ['input_ids', 'input_mask', 'input_type_ids']

            output = tf.import_graph_def(graph_def,
                                         input_map={k + ':0': features[k] for k in input_names},
                                         return_elements=['final_encodes:0'])
            return EstimatorSpec(mode=mode, predictions={
                # 'client_id': features['client_id'],
                'encodes': output[0]
            })

        config = tf.ConfigProto(device_count={'GPU': 0 if self.device_id < 0 else 1})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        # session-wise XLA doesn't seem to work on tf 1.10
        # if args.xla:
        #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))

    def start_encoding(self, msg):
        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compatibility

        tf = import_tf(self.device_id, self.verbose, use_fp16=self.use_fp16)
        estimator = self.get_estimator(tf)
        vectors = []
        for r in estimator.predict(self.input_fn_builder(msg, tf), yield_single_examples=False):
            vectors.append(r['encodes'])
        return vectors

    def input_fn_builder(self, msg, tf):
        def gen():
            tokenizer = FullTokenizer(vocab_file=os.path.join(self.model_dir, 'vocab.txt'))
            # Windows does not support logger in MP environment, thus get a new logger
            # inside the process for better compatibility

            print('ready and listening!')

            # check if msg is a list of list, if yes consider the input is already tokenized
            is_tokenized = all(isinstance(el, list) for el in msg)
            tmp_f = list(convert_lst_to_features(msg, self.max_seq_len,
                                                 self.bert_config.max_position_embeddings,
                                                 tokenizer,
                                                 is_tokenized, self.mask_cls_sep))
            result = {
                # 'client_id': 'aaa',
                'input_ids': [f.input_ids for f in tmp_f],
                'input_mask': [f.input_mask for f in tmp_f],
                'input_type_ids': [f.input_type_ids for f in tmp_f]
            }
            yield result

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={
                    # 'client_id': tf.string,
                    'input_ids': tf.int32,
                    'input_mask': tf.int32,
                    'input_type_ids': tf.int32},
                output_shapes={
                    # 'client_id': (),
                    'input_ids': (None, None),
                    'input_mask': (None, None),
                    'input_type_ids': (None, None)}).prefetch(self.prefetch_size))

        return input_fn


class BertServer:
    def __init__(self, args):
        self.model_dir = args.model_dir
        self.max_seq_len = args.max_seq_len
        self.num_worker = args.num_worker
        self.max_batch_size = args.max_batch_size
        self.args = args
        print('freeze, optimize and export graph, could take a while...')
        # with Pool(processes=1) as pool:
            # optimize the graph, must be done in another process
            # self.graph_path, self.bert_config = pool.apply(optimize_graph, (self.args,))
        self.graph_path, self.bert_config = optimize_graph(args)
        # from .graph import optimize_graph
        # self.graph_path = optimize_graph(self.args, self.logger)
        if self.graph_path:
            print('optimized graph is stored at: %s' % self.graph_path)
        else:
            raise FileNotFoundError('graph optimization fails and returns empty result')

    def encode(self, msg):
        process = BertWorker(self.args, self.graph_path, self.bert_config)
        vector = process.start_encoding(msg)
        return vector


def import_tf(device_id=-1, verbose=False, use_fp16=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf


def optimize_graph(args, logger=None):
    try:
        # we don't need GPU for optimizing the graph
        tf = import_tf(verbose=args.verbose)
        from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

        config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)

        config_fp = os.path.join(args.model_dir, args.config_name)
        init_checkpoint = os.path.join(args.tuned_model_dir or args.model_dir, args.ckpt_name)
        if args.fp16:
            print('fp16 is turned on! '
                  'Note that not all CPU GPU support fast fp16 instructions, '
                  'worst case you will have degraded performance!')
        print('model config: %s' % config_fp)
        print(
            'checkpoint%s: %s' % (
                ' (override by the fine-tuned model)' if args.tuned_model_dir else '', init_checkpoint))
        with tf.gfile.GFile(config_fp, 'r') as f:
            bert_config = modeling.BertConfig.from_dict(json.load(f))

        print('build graph...')
        # input placeholders, not sure if they are friendly to XLA
        input_ids = tf.placeholder(tf.int32, (None, None), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, None), 'input_mask')
        input_type_ids = tf.placeholder(tf.int32, (None, None), 'input_type_ids')

        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope if args.xla else contextlib.suppress

        with jit_scope():
            input_tensors = [input_ids, input_mask, input_type_ids]

            model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids,
                use_one_hot_embeddings=False)

            tvars = tf.trainable_variables()

            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            minus_mask = lambda x, m: x - tf.expand_dims(1.0 - m, axis=-1) * 1e30
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_max = lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

            with tf.variable_scope("pooling"):
                if len(args.pooling_layer) == 1:
                    encoder_layer = model.all_encoder_layers[args.pooling_layer[0]]
                else:
                    all_layers = [model.all_encoder_layers[l] for l in args.pooling_layer]
                    encoder_layer = tf.concat(all_layers, -1)

                input_mask = tf.cast(input_mask, tf.float32)
                if args.pooling_strategy == PoolingStrategy.REDUCE_MEAN:
                    pooled = masked_reduce_mean(encoder_layer, input_mask)
                elif args.pooling_strategy == PoolingStrategy.REDUCE_MAX:
                    pooled = masked_reduce_max(encoder_layer, input_mask)
                elif args.pooling_strategy == PoolingStrategy.REDUCE_MEAN_MAX:
                    pooled = tf.concat([masked_reduce_mean(encoder_layer, input_mask),
                                        masked_reduce_max(encoder_layer, input_mask)], axis=1)
                elif args.pooling_strategy == PoolingStrategy.FIRST_TOKEN or \
                        args.pooling_strategy == PoolingStrategy.CLS_TOKEN:
                    pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
                elif args.pooling_strategy == PoolingStrategy.LAST_TOKEN or \
                        args.pooling_strategy == PoolingStrategy.SEP_TOKEN:
                    seq_len = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.int32)
                    rng = tf.range(0, tf.shape(seq_len)[0])
                    indexes = tf.stack([rng, seq_len - 1], 1)
                    pooled = tf.gather_nd(encoder_layer, indexes)
                elif args.pooling_strategy == PoolingStrategy.NONE:
                    pooled = mul_mask(encoder_layer, input_mask)
                else:
                    raise NotImplementedError()

            if args.fp16:
                pooled = tf.cast(pooled, tf.float16)

            pooled = tf.identity(pooled, 'final_encodes')
            output_tensors = [pooled]
            tmp_g = tf.get_default_graph().as_graph_def()

        with tf.Session(config=config) as sess:
            print('load parameters from checkpoint...')

            sess.run(tf.global_variables_initializer())
            dtypes = [n.dtype for n in input_tensors]
            print('optimize...')
            tmp_g = optimize_for_inference(
                tmp_g,
                [n.name[:-2] for n in input_tensors],
                [n.name[:-2] for n in output_tensors],
                [dtype.as_datatype_enum for dtype in dtypes],
                False)

            print('freeze...')
            tmp_g = convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors],
                                                   use_fp16=args.fp16)

        tmp_file = tempfile.NamedTemporaryFile('w', delete=False, dir=args.graph_tmp_dir).name
        print('write graph to a tmp file: %s' % tmp_file)
        with tf.gfile.GFile(tmp_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return tmp_file, bert_config
    except Exception:
        raise Exception('fail to optimize the graph!', exc_info=True)


def convert_variables_to_constants(sess,
                                   input_graph_def,
                                   output_node_names,
                                   variable_names_whitelist=None,
                                   variable_names_blacklist=None,
                                   use_fp16=False):
    from tensorflow.python.framework.graph_util_impl import extract_sub_graph
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.framework import node_def_pb2
    from tensorflow.core.framework import attr_value_pb2
    from tensorflow.core.framework import types_pb2
    from tensorflow.python.framework import tensor_util

    def patch_dtype(input_node, field_name, output_node):
        if use_fp16 and (field_name in input_node.attr) and (input_node.attr[field_name].type == types_pb2.DT_FLOAT):
            output_node.attr[field_name].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_HALF))

    inference_graph = extract_sub_graph(input_graph_def, output_node_names)

    variable_names = []
    variable_dict_names = []
    for node in inference_graph.node:
        if node.op in ["Variable", "VariableV2", "VarHandleOp"]:
            variable_name = node.name
            if ((variable_names_whitelist is not None and
                 variable_name not in variable_names_whitelist) or
                    (variable_names_blacklist is not None and
                     variable_name in variable_names_blacklist)):
                continue
            variable_dict_names.append(variable_name)
            if node.op == "VarHandleOp":
                variable_names.append(variable_name + "/Read/ReadVariableOp:0")
            else:
                variable_names.append(variable_name + ":0")
    if variable_names:
        returned_variables = sess.run(variable_names)
    else:
        returned_variables = []
    found_variables = dict(zip(variable_dict_names, returned_variables))

    output_graph_def = graph_pb2.GraphDef()
    how_many_converted = 0
    for input_node in inference_graph.node:
        output_node = node_def_pb2.NodeDef()
        if input_node.name in found_variables:
            output_node.op = "Const"
            output_node.name = input_node.name
            dtype = input_node.attr["dtype"]
            data = found_variables[input_node.name]

            if use_fp16 and dtype.type == types_pb2.DT_FLOAT:
                output_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(data.astype('float16'),
                                                             dtype=types_pb2.DT_HALF,
                                                             shape=data.shape)))
            else:
                output_node.attr["dtype"].CopyFrom(dtype)
                output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(data, dtype=dtype.type,
                                                         shape=data.shape)))
            how_many_converted += 1
        elif input_node.op == "ReadVariableOp" and (input_node.input[0] in found_variables):
            # placeholder nodes
            # print('- %s | %s ' % (input_node.name, input_node.attr["dtype"]))
            output_node.op = "Identity"
            output_node.name = input_node.name
            output_node.input.extend([input_node.input[0]])
            output_node.attr["T"].CopyFrom(input_node.attr["dtype"])
            if "_class" in input_node.attr:
                output_node.attr["_class"].CopyFrom(input_node.attr["_class"])
        else:
            # mostly op nodes
            output_node.CopyFrom(input_node)

        patch_dtype(input_node, 'dtype', output_node)
        patch_dtype(input_node, 'T', output_node)
        patch_dtype(input_node, 'DstT', output_node)
        patch_dtype(input_node, 'SrcT', output_node)
        patch_dtype(input_node, 'Tparams', output_node)

        if use_fp16 and ('value' in output_node.attr) and (
                output_node.attr['value'].tensor.dtype == types_pb2.DT_FLOAT):
            # hard-coded value need to be converted as well
            output_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
                tensor=tensor_util.make_tensor_proto(
                    output_node.attr['value'].tensor.float_val[0],
                    dtype=types_pb2.DT_HALF)))

        output_graph_def.node.extend([output_node])

    output_graph_def.library.CopyFrom(inference_graph.library)
    return output_graph_def


args = Arguments()
server = BertServer(args)


def sentence_to_vector(sentences):
    result_vector = server.encode(sentences)
    return result_vector[0]


def build_news_vec():
    """
    构建新闻向量和对应id_index字典
    :return:
    """
    news_file = 'dev/news.tsv'
    col_spliter = '\t'
    news_titles = []
    index = 0
    news_id_index = {}
    with tf.io.gfile.GFile(news_file, "r") as rd:
        for line in rd:
            nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(col_spliter)
            news_id_index[nid] = index
            index += 1
            news_titles.append(title)
    result = sentence_to_vector(news_titles)
    np.save('dev/dev_news.npy', result)
    np.save('dev/dev_news_id_index.npy', news_id_index)


# news_vec = np.load('../data_demo/train/train_news.npy')
# news_id_index = np.load('../data_demo/train/train_news_id_index.npy', allow_pickle=True).item()


if __name__ == "__main__":
    # sentences = ["White House: Trump, Kim summit ends without reaching a deal",
    #              "Second Trump-Kim Summit Ends With No Deal",
    #              "UN consultant among Indians killed in Ethiopian plane crash: Sushma Swaraj",
    #              "Atleast a Dozen UN-affiliated Personnel Killed in Ethiopian Crash: UN Source",
    #              "Donald Trump says hopefully India, Pakistan conflict coming to an end"]
    # sentences = ['today is very hot', 'It is so hot today']
    #
    # result = sentence_to_vector(sentences)
    # print(result)
    build_news_vec()

