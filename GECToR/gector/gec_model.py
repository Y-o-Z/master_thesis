"""
Copyright (c) 2020 Grammarly - GECToR (https://github.com/grammarly/gector)  [adapted by Yoel Zweig]
"""

import time
import logging
import copy

import torch
from allennlp.data.tokenizers import Token  # simple Tokenizer
from allennlp.data.fields import TextField  # → list of string tokens
from allennlp.data.instance import Instance  # → collection of fields
from allennlp.data.dataset import Batch  # → collection of instances
from allennlp.data.vocabulary import Vocabulary  # interface
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder  # embeds output from a TokenIndexer
from allennlp.nn import util

from gector.wordpiece_indexer import PretrainedBertIndexer  # create subword tokens
from gector.bert_token_embedder import PretrainedBertEmbedder  # create embeddings for first subword token of words
from gector.seq2labels_model import Seq2Labels  # model
# get_target_sent_by_edits => apply edits to source_tokens => transform
from g_utils.helpers import START_TOKEN, UNK, PAD, get_target_sent_by_edits, get_weights_name

from t_helper import get_g_transformations, is_append, is_replace, is_delete, is_merge_space, is_split_hyphen

logging.getLogger("werkzeug").setLevel(logging.ERROR)  # name
logger = logging.getLogger(__file__)  # __file__ -> pathname of file from which module was loaded

g_transformations = get_g_transformations()

# _..    -> internal use indicator -> not imported by from x import *
# .._    -> avoid keyword conflicts
# __..   -> class stuff
# __..__ -> special variables


class GecBERTModel(object):
    def __init__(self, model_paths=None, vocab_path=None,
                 max_len=50,
                 min_len=3,
                 lowercase_tokens=False,
                 model_name='bert',
                 iterations=5,
                 confidence=0.0,
                 min_confidence_probability=0.0,
                 special_tokens_fix=0,
                 is_ensemble=False,
                 weights=None,
                 resolve_cycles=False,
                 log=False):
        self.model_weights = list(map(float, weights)) if weights else [1] * len(model_paths)  # default = uniform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vocab = Vocabulary.from_files(vocab_path)
        self.max_len = max_len
        self.min_len = min_len
        self.lowercase_tokens = lowercase_tokens
        self.iterations = iterations
        self.confidence = confidence
        self.min_confidence_probability = min_confidence_probability
        self.resolve_cycles = resolve_cycles
        self.log = log

        self.indexers = []
        self.models = []
        for model_path in model_paths:
            if is_ensemble:
                model_name, special_tokens_fix = self._get_model_data(model_path)
            weights_name = get_weights_name(model_name, lowercase_tokens)
            # list of indexers, one per model
            self.indexers.append(self._get_indexer(weights_name, special_tokens_fix))
            model = Seq2Labels(vocab=self.vocab,
                               text_field_embedder=self._get_embbeder(weights_name, special_tokens_fix),
                               confidence=self.confidence
                               ).to(self.device)
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_path))
            else:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            self.models.append(model)

    def predict(self, batches):
        t_before = time.time()
        predictions = []
        for batch, model in zip(batches, self.models):
            batch = util.move_to_device(batch.as_tensor_dict(), 0 if torch.cuda.is_available() else -1)
            with torch.no_grad():
                prediction = model.forward(**batch)
            predictions.append(prediction)

        preds, idx, error_probs = self._convert(predictions)  # choose max likelihood predictions
        t_after = time.time()
        if self.log:
            print(f"Inference time {t_after - t_before}")
        return preds, idx, error_probs

    @staticmethod
    def _get_model_data(model_path):
        model_name = model_path.split('/')[-1]
        tr_model, stf = model_name.split('_')[:2]  # special tokens fix
        return tr_model, int(stf)

    def _get_indexer(self, weights_name, special_tokens_fix):  # PretrainedBertIndexer
        bert_token_indexer = PretrainedBertIndexer(
            pretrained_model=weights_name,
            do_lowercase=self.lowercase_tokens,
            max_pieces_per_token=5,
            use_starting_offsets=True,
            truncate_long_sequences=True,
            special_tokens_fix=special_tokens_fix,
            is_test=True
        )
        return {'bert': bert_token_indexer}

    @staticmethod
    def _get_embbeder(weights_name, special_tokens_fix):  # PretrainedBertEmbedder → BasicTextFieldEmbedder
        embedders = {'bert': PretrainedBertEmbedder(
            pretrained_model=weights_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=special_tokens_fix)
        }
        text_field_embedder = BasicTextFieldEmbedder(
            token_embedders=embedders,
            embedder_to_indexer_map={"bert": ["bert", "bert-offsets"]},
            allow_unmatched_keys=True)
        return text_field_embedder

    def _convert(self, data):
        all_class_probs = torch.zeros_like(data[0]['class_probabilities_labels'])
        error_probs = torch.zeros_like(data[0]['max_error_probability'])
        for output, weight in zip(data, self.model_weights):
            all_class_probs += weight * output['class_probabilities_labels'] / sum(self.model_weights)
            error_probs += weight * output['max_error_probability'] / sum(self.model_weights)

        max_vals = torch.max(all_class_probs, dim=-1)
        probs = max_vals[0].tolist()
        idx = max_vals[1].tolist()
        return probs, idx, error_probs.tolist()

    def get_token_action(self, index, prob, sugg_token):
        # do nothing
        if prob < self.min_confidence_probability or sugg_token in [UNK, PAD, '$KEEP']:
            return None

        if sugg_token.startswith('$REPLACE_') or sugg_token.startswith('$TRANSFORM_') or sugg_token == '$DELETE':
            start_pos = index
            end_pos = index + 1
        elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
            start_pos = index + 1
            end_pos = index + 1

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index('_') + 1:]

        return start_pos - 1, end_pos - 1, sugg_token_clear, sugg_token

    def preprocess(self, token_batch):  # e.g. token_batch = [["I", "have", "always", ".."], ...]
        seq_lens = [len(sequence) for sequence in token_batch if sequence]  # skip empty entries
        if not seq_lens:
            return []
        max_len = min(max(seq_lens), self.max_len)
        batches = []  # one batch for each indexer
        for indexer in self.indexers:
            batch = []
            for sequence in token_batch:
                tokens = sequence[:max_len]  # cut off sentence tails
                # Token(x) -> simple token representation of x, \approx ordered dictionary
                tokens = [Token(token) for token in ['$START'] + tokens]
                batch.append(Instance({'tokens': TextField(tokens, indexer)}))
            batch = Batch(batch)  # collection of instances
            batch.index_instances(self.vocab)  # string -> int (map)
            batches.append(batch)

        return batches

    def postprocess_batch(self, batch, all_probabilities, all_ids, correctness_probs):
        all_results, all_transformations, all_indices = [], [], []
        # noop_index = self.vocab.get_token_index("$KEEP", namespace="labels")  # 0 (no operation)
        for index, (tokens, probabilities, ids, correctness_prob) in \
                enumerate(zip(batch, all_probabilities, all_ids, correctness_probs)):
            length = min(len(tokens), self.max_len)  # <= 50, no changes for tail
            edits, transformations, indices = [], [], []

            # skip whole sentence if there are no errors (all $KEEP (0)) or if probability of correctness is too low
            if not any(ids) or correctness_prob < self.min_confidence_probability:
                all_results.append(tokens)
                continue

            for i in range(length + 1):
                # skip if there is no error
                if not ids[i]:  # == noop_index:
                    transformations.append("$KEEP")
                    continue

                sugg_token = self.vocab.get_token_from_index(ids[i], namespace='labels')
                # action = start index, stop index, sugg_token_clear, sugg_token
                action = self.get_token_action(i, probabilities[i], sugg_token)  # sugg_token = label

                if not action:
                    transformations.append("$KEEP")  # UNK, ...
                    continue

                edits.append(action[:3])  # start index, stop index, sugg_token_clear
                transformations.append(action[3])  # sugg_token = label
                indices.append(i)

            all_results.append(get_target_sent_by_edits(tokens, edits))
            all_transformations.append(transformations)
            all_indices.append((index, indices))

        return all_results, all_transformations, all_indices

    def update_final_batch(self, final_batch, pred_ids, pred_batch, prev_preds_dict):
        # prev_pred_dict → numbered final_batch
        new_pred_ids = []
        total_updated = 0
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]
            pred = pred_batch[i]
            prev_preds = prev_preds_dict[orig_id]
            if orig != pred and pred not in prev_preds:
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1
            elif orig != pred and pred in prev_preds:
                # update final batch, but stop iterations
                final_batch[orig_id] = pred
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated

    def handle_batch(self, full_batch):
        """
        Apply basic and grammatical transformations/corrections to batch of sentences
        """
        final_batch = full_batch[:]  # shallow copy
        sentences_dict = {i: [final_batch[i]] for i in range(len(final_batch))}  # {0: [sen1], 1: [sen2], ...}
        short_ids = [i for i in range(len(final_batch)) if len(final_batch[i]) < self.min_len]  # [0,1,..] or [] if >=
        pred_ids = [i for i in range(len(final_batch)) if i not in short_ids]  # \approx ids - short_ids
        total_updates = 0

        for _ in range(self.iterations):
            batch = [final_batch[i] for i in pred_ids]  # list of token lists

            sequences = self.preprocess(batch)  # allennlp.data.dataset.Batch (+ skip empty, cut of tails)

            if not sequences:
                break

            # one entry per sentence (→ pred_ids)
            # probabilities → confidence in predicted labels []
            # idxs → predicted label indices ($KEEP, $DELETE, ...) []
            # error_probs → confidence in prediction (overestimated)
            probabilities, idxs, error_probs = self.predict(sequences)

            pred_batch, _, _ = self.postprocess_batch(batch, probabilities, idxs, error_probs)

            final_batch, pred_ids, cnt = self.update_final_batch(final_batch, pred_ids, pred_batch, sentences_dict)
            total_updates += cnt

            if not pred_ids:
                break

        return final_batch, total_updates

    def divide_by_error_classes(self, full_batch):
        """
        :param full_batch: batch of sentences (<= 32)
        :return
        - correct_sentences: list of (non-unique) corrected sentences that originally contained
        at least one error belonging to a grammatical category (→ g-transformations)

        - sentences_with_one_error: corresponding list of (unique) sentences with exactly one of these
        grammatical category errors (→ g-transformations)

        - error_dicts: corresponding list of dictionaries with error class and indices describing
        the position(s) of the correct and incorrect token(s)

        (len(correct_sentences) == len(sentences_with_one_error) == len(error_dicts))
        """
        final_batch = full_batch[:]
        sentences_dict = {i: [final_batch[i]] for i in range(len(final_batch))}  # number sentences
        short_ids = [i for i in range(len(final_batch)) if len(final_batch[i]) < self.min_len]  # ignore indices
        pred_ids = [i for i in range(len(final_batch)) if i not in short_ids]  # predict indices
        final_batch_g_transformations = [[[] for _ in range(len(final_batch))]]  # one list per iteration

        for n_iter in range(self.iterations):
            batch = [final_batch[i] for i in pred_ids]

            sequences = self.preprocess(batch)

            if not sequences:
                break

            probabilities, idxs, error_probs = self.predict(sequences)

            pred_batch, pred_transformations, change_indices = self.postprocess_batch(batch, probabilities,
                                                                                      idxs, error_probs)

            # keep track of grammatical corrections
            for pos, (sentence_number, indices) in enumerate(change_indices):
                if not indices:  # no changes
                    continue
                transformations = pred_transformations[pos]
                for index in indices:
                    if transformations[index] in g_transformations:
                        try:
                            d = {"label": transformations[index], "index": index - 1}  # - 1 because of $START
                            if is_merge_space(transformations[index]):
                                d["prev_form"] = f"{batch[sentence_number][index - 1]} {batch[sentence_number][index]}"
                            else:
                                d["prev_form"] = f"{batch[sentence_number][index - 1]}"
                            final_batch_g_transformations[n_iter][pred_ids[sentence_number]].append(d)
                        except:
                            print("ERROR appending to final_batch_g_transformations: (1)")
                    # also track $APPEND, $REPLACE, $DELETE → overwrite or change indices
                    elif transformations[index] != "$KEEP":
                        try:
                            d = {"label": transformations[index], "index": index - 1}  # - 1 because of $START
                            final_batch_g_transformations[n_iter][pred_ids[sentence_number]].append(d)
                        except:
                            print("ERROR appending to final_batch_g_transformations: (2)")
            final_batch_g_transformations.append([[] for _ in range(len(full_batch))])

            final_batch, pred_ids, _ = self.update_final_batch(final_batch, pred_ids, pred_batch, sentences_dict)

            if not pred_ids:
                break

        # build triplets → correct_sentences, sentences_with_one_error, error_dicts
        # based on applied transformations
        correct_sentences = []
        sentences_with_one_error = []
        error_dicts = []
        for sentence_id in range(len(final_batch)):  # for each sentence
            transformations = []
            for n_iter in range(len(final_batch_g_transformations)):  # over all iterations
                current_transformations = []
                if not any(final_batch_g_transformations[n_iter][sentence_id]):  # no transformations
                    continue
                # add all transformations (from current iteration)
                for index in range(len(final_batch_g_transformations[n_iter][sentence_id])):
                    current_transformations.append(copy.deepcopy(final_batch_g_transformations[n_iter][sentence_id][index]))
                # is there an overwrite for an (older) existing transformation?
                keep_indices = [1] * len(transformations)
                for index in range(len(transformations)):
                    for current_transformation in current_transformations:
                        if current_transformation["index"] == transformations[index]["index"] and not \
                                is_append(current_transformation["label"]):
                            keep_indices[index] = 0
                            break
                transformations = [t for i, t in enumerate(transformations) if keep_indices[i]]
                # how do older transformations change
                for index in range(len(transformations)):
                    index_shift = 0
                    for current_transformation in current_transformations:
                        if current_transformation["index"] < transformations[index]["index"]:
                            if is_append(current_transformation["label"]):
                                index_shift += 1
                            elif is_delete(current_transformation["label"]):
                                index_shift -= 1
                            elif is_merge_space(current_transformation["label"]):
                                index_shift -= 1
                            elif is_split_hyphen(current_transformation["label"]):
                                index_shift += 1
                    transformations[index]["index"] += index_shift
                # go through transformations in reverse (later indices (positions) to earlier indices)
                for index in range(len(current_transformations)):
                    index_shift = 0
                    for other in range(index - 1, -1, -1):
                        if is_append(current_transformations[other]["label"]):
                            index_shift += 1
                        elif is_delete(current_transformations[other]["label"]):
                            index_shift -= 1
                        elif is_merge_space(current_transformations[other]["label"]):
                            index_shift -= 1
                        elif is_split_hyphen(current_transformations[other]["label"]):
                            index_shift += 1
                    current_transformations[index]["index"] += index_shift
                # get rid of $APPEND, $REPLACE, $DELETE
                current_transformations = [t for t in current_transformations if not (is_delete(t["label"]) or
                                                                                      is_append(t["label"]) or
                                                                                      is_replace(t["label"]))]
                transformations.extend(current_transformations)

            correct_sentence = " ".join(final_batch[sentence_id])
            try:
                for j in range(len(transformations)):
                    index = transformations[j]["index"]
                    label = transformations[j]["label"]
                    error_class = g_transformations[label]
                    error_sentence = final_batch[sentence_id][:]
                    # undo g-transformation
                    error_sentence[index] = copy.deepcopy(transformations[j]["prev_form"])
                    c_index = [index, index]  # c → correct
                    i_index = [index, index]  # i → incorrect
                    if is_merge_space(label):
                        i_index[1] += 1  # 2 tokens in incorrect sentence
                    if is_split_hyphen(label):
                        c_index[1] += 1  # 2 tokens in correct sentence
                        # error_sentence[error_sentence[:transformations[j]["index"] + 1]].pop()
                    error_sentence = " ".join(error_sentence)

                    correct_sentences.append(correct_sentence)
                    sentences_with_one_error.append(error_sentence)
                    error_dicts.append({"error_class": error_class, "preds_indices": c_index,
                                        "one_error_preds_indices": i_index})
                    # print(correct_sentence)
                    # print(error_sentence)
                    # print({"error_class": error_class, "preds_indices": c_index, "one_error_preds_indices": i_index})
                    # print("**********")
                    # _ = input()
            except:
                print("ERROR in constructing triplet")

        return correct_sentences, sentences_with_one_error, error_dicts
