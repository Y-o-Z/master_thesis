"""
Copyright (c) 2020 Grammarly - GECToR (https://github.com/grammarly/gector)  [adapted by Yoel Zweig]
"""

import argparse

from gector.gec_model import GecBERTModel

from t_students import NUM_ERROR_CLASSES
from t_helper import write_pickle, read_lines  # is used


def predict_for_file(input_file, output_file, model, batch_size=32):
    test_data = read_lines(input_file)  # drop empty lines and trailing whitespace
    predictions = []
    cnt_corrections = 0
    batch = []
    for sentence in test_data:
        batch.append(sentence.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)  # list of tokens, num_corrections
            predictions.extend(preds)  # append elements from iterable
            cnt_corrections += cnt
            batch = []
    if batch:  # last batch nonempty?, len(batch) <= batch_size
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    with open(output_file, 'w') as file:
        file.write("\n".join([" ".join(x) for x in predictions]))

    return cnt_corrections


def divide_by_error_classes(input_file, model, batch_size=32, dataset="conll", folder="z_test/"):
    """
    Divide sentences into n=NUM_ERROR_CLASSES error classes, one pickle file per error class
    with triplets [correct_sentence, sentence_with_one_error, error_dict]
    """
    data = read_lines(input_file)  # drop empty lines and trailing whitespace
    triplet_list = []
    batch = []
    batch_count = 0
    for sentence in data:
        batch.append(sentence.split())
        if len(batch) == batch_size:
            print(f"### batch = {batch_count} ###")
            batch_count += 1
            # preds → list of corrected sentences (not unique)
            # one_error_preds → corresponding list of corrected sentences with one ERROR_CLASS error (unique)
            # error_dicts → corresponding list of {error_class, preds_indices [i, j], one_error_preds_indices [i2, j2]}
            preds, one_error_preds, error_dicts = model.divide_by_error_classes(batch)
            for i in range(len(preds)):
                triplet_list.append([preds[i], one_error_preds[i], error_dicts[i]])
                if i % 5 == 0 and False:
                    print(preds[i], one_error_preds[i], error_dicts[i], sep="\n", end="\n"+"-"*10+"\n")
            batch = []
    if batch:  # last batch nonempty?, len(batch) <= batch_size
        print(f"### (last) batch = {batch_count} ###")
        preds, one_error_preds, error_dicts = model.divide_by_error_classes(batch)
        for i in range(len(preds)):
            triplet_list.append([preds[i], one_error_preds[i], error_dicts[i]])

    for error_class in range(NUM_ERROR_CLASSES):
        exec(f"list_{dataset}_{error_class} = []")  # one list per error class

    for pred, one_error_pred, error_dict in triplet_list:
        try:
            exec(f"list_{dataset}_{error_dict['error_class']}.append([pred, one_error_pred, error_dict])")
        except:
            print(f"error in list_{dataset}_{error_dict['error_class']}.append(...)")

    for error_class in range(NUM_ERROR_CLASSES):
        name = f"list_{dataset}_{error_class}"
        exec(f"write_pickle(list_{dataset}_{error_class}, folder + name)")


def main(args):
    if not args.exercises:
        model = GecBERTModel(model_paths=args.model_path,
                             vocab_path=args.vocab_path,
                             max_len=args.max_len,
                             min_len=args.min_len,
                             lowercase_tokens=args.lowercase_tokens,
                             model_name=args.transformer_model,
                             iterations=args.iteration_count,
                             confidence=args.additional_confidence,  # positive bias for $KEEP
                             min_confidence_probability=args.min_confidence_probability,  # $KEEP if lower
                             special_tokens_fix=args.special_tokens_fix,  # → indexer, embedder
                             is_ensemble=args.is_ensemble,
                             weights=args.weights,
                             log=False)

        cnt_corrections = predict_for_file(args.input_file, args.output_file, model, batch_size=args.batch_size)

        print(f"Produced overall corrections: {cnt_corrections}")
    else:
        model = get_single_model()
        divide_by_error_classes(args.input_file, model)


def get_single_model(model="xlnet_0_gector.th", model_name="xlnet", confidence=.35, min_confidence_probability=.66):
    s1 = [f"g_trained_models/{model}"]
    s2 = "data/output_vocabulary/"
    model = GecBERTModel(model_paths=s1,
                         vocab_path=s2,
                         max_len=50,
                         min_len=3,
                         lowercase_tokens=False,
                         model_name=model_name,
                         iterations=5,
                         confidence=confidence,
                         min_confidence_probability=min_confidence_probability,
                         special_tokens_fix=0,
                         is_ensemble=0,
                         weights=None,
                         log=False)
    return model


def get_multi_model(model_1="bert_0_gector.th", model_2="roberta_1_gector.th", model_3="xlnet_0_gector.th"):
    s1 = [f"g_trained_models/{model_1}",
          f"g_trained_models/{model_2}",
          f"g_trained_models/{model_3}"]
    s2 = "data/output_vocabulary/"
    model = GecBERTModel(model_paths=s1,
                         vocab_path=s2,
                         max_len=50,
                         min_len=3,
                         lowercase_tokens=False,
                         model_name="",
                         iterations=5,
                         confidence=.16,
                         min_confidence_probability=.4,
                         special_tokens_fix=0,
                         is_ensemble=1,
                         weights=None,
                         log=False)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path',
                        nargs='+',  # args into list, error if empty
                        help='Path to the model file',
                        required=True)
    parser.add_argument('-v', '--vocab_path',
                        help='Path to the vocabulary (labels)',
                        default='data/output_vocabulary')  # → size trade-off
    parser.add_argument('-i', '--input_file',
                        help='Path to the input file',
                        required=True)
    parser.add_argument('-o', '--output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--max_len',  # preprocess()
                        type=int,
                        help='The maximum sentence length '
                             '(longer sentences will be truncated i.e. the tails will be returned w/o changes)',
                        default=50)
    parser.add_argument('--min_len',  # else not included in pred_ids
                        type=int,
                        help='The minimum sentence length '
                             '(shorter sentences will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of the hidden unit cell',  # lines (sentences) per iteration
                        default=32)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens (or not)',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'xlnet', 'roberta'],  # + gpt2 ..
                        help='Name of the transformer model',
                        default='bert')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How much probability to add to $KEEP token',
                        default=0.0)
    parser.add_argument('--min_confidence_probability',  # → .postprocess_batch()
                        type=float,
                        help='Lower probability threshold for sentence predictions '
                             'and for individual token predictions',
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization (or not). '  # [$START]
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa',
                        default=0)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling (or not)',
                        default=0)
    parser.add_argument('--weights',
                        nargs='+',
                        help='Used to calculate weighted average',  # for ensemble
                        default=None)
    parser.add_argument('--exercises',
                        type=int,
                        help='Whether to create some exercises (or not)',
                        default=0)
    args = parser.parse_args()
    main(args)
