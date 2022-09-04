import argparse
import json
import numpy as np
import os
import pickle
import random
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Generator, List, Optional, Tuple, Union
import constant
from models.transformer_box_model_v1 import TransformerBoxModelV1
from data_utils import DatasetLoader
from data_utils import to_torch
from modules.box_decoder import get_classifier

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", help = "Identifier for model")
parser.add_argument("--device", type = int, default = 0, help = "CUDA device")
parser.add_argument("--n_gpu", help = "Number of GPUs.", type = int, default = 1)
parser.add_argument("--mode",
                    help = "Whether to train or test",
                    default = "train",
                    choices = ["train", "test"])
parser.add_argument("--local_rank",
                    type = int,
                    default = -1,
                    help = "For distributed training: local_rank")

# Data
parser.add_argument("--train_data", help = "Train data file pattern.", default = "")
parser.add_argument("--dev_data", help = "Dev data filename.", default = "")
parser.add_argument("--eval_data", help = "Test data filename.", default = "")
parser.add_argument("--goal",
                    help = "category vocab size.",
                    default = "60k",
                    choices = ["60k", "ufet", "5k", "1k", "onto", "figer",
                               "bbn", "toy"])
parser.add_argument("--seed", help = "Random Seed", type = int, default = 0)
parser.add_argument("--context_window_size",
                    help = "Left and right context size.",
                    default = 100)

# learning
parser.add_argument("--num_epoch",
                    help = "The number of epoch.",
                    default = 100,
                    type = int)
parser.add_argument("--per_gpu_train_batch_size",
                    help = "The batch size per GPU",
                    default = 8,
                    type = int)
parser.add_argument("--per_gpu_eval_batch_size",
                    help = "The batch size per GPU.",
                    default = 8,
                    type = int)
parser.add_argument("--learning_rate_enc",
                    help = "BERT: start learning rate.",
                    default = 2e-5,
                    type = float)
parser.add_argument("--learning_rate_cls",
                    help = "Classifier: start learning rate.",
                    default = 1e-3,
                    type = float)
parser.add_argument("--adam_epsilon_enc",
                    default = 1e-8,
                    type = float,
                    help = "BERT: Epsilon for Adam optimizer.")
parser.add_argument("--adam_epsilon_cls",
                    default = 1e-8,
                    type = float,
                    help = "Classifier: Epsilon for Adam optimizer.")
parser.add_argument("--hidden_dropout_prob",
                    help = "Dropout rate",
                    default = .0,
                    type = float)
parser.add_argument("--n_negatives",
                    help = "num of negative types during training.",
                    default = 0,
                    type = int)
parser.add_argument("--neg_temp",
                    help = "Temperature for softmax over negatives.",
                    default = .0,
                    type = float)
parser.add_argument("--gradient_accumulation_steps",
                    help = "Number of updates steps to accumulate before "
                           "performing a backward/update pass.",
                    default = 1,
                    type = int)
parser.add_argument("--alpha_l2_reg_cls",
                    help = "Coefficient for L2 regularization on classifier "
                           "parameters.",
                    default = -1.,
                    type = float)
# -- Learning rate scheduler.
parser.add_argument("--use_scheduler_enc",
                    help = "BERT: use lr scheduler.",
                    action = 'store_true')
parser.add_argument("--use_scheduler_cls",
                    help = "Classifier: use lr scheduler.",
                    action = 'store_true')
parser.add_argument("--scheduler_type", help = "Scheduler type.",
                    default = "cyclic")
parser.add_argument("--step_size_up",
                    help = "Step size up.",
                    default = 2000,
                    type = int)
parser.add_argument("--warmup_steps",
                    help = "Linear warmup over warmup_steps.",
                    default = 0,
                    type = int)
parser.add_argument("--max_steps",
                    default = 1000000,
                    type = int,
                    help = "max # of steps.")

# Model
parser.add_argument("--model_type",
                    default = "bert-large-uncased-whole-word-masking",
                    choices = [
                        "bert-base-uncased",
                        "bert-large-uncased",
                        "bert-large-uncased-whole-word-masking",
                        "roberta-base",
                        "roberta-large"
                    ])
parser.add_argument("--emb_type",
                    help = "Embedding type.",
                    default = "baseline",
                    choices = ["baseline", "box"])
parser.add_argument("--mc_box_type",
                    help = "Mention&Context box type.",
                    default = "SigmoidBoxTensor")
parser.add_argument("--type_box_type",
                    help = "Type box type.",
                    default = "SigmoidBoxTensor")
parser.add_argument("--box_dim", help = "Box dimension.", default = 200, type = int)
parser.add_argument("--box_offset",
                    help = "Offset for constant box.",
                    default = 0.5,
                    type = float)
parser.add_argument("--threshold",
                    help = "Threshold for type classification.",
                    default = 0.5,
                    type = float)
parser.add_argument("--alpha_type_reg",
                    help = "alpha_type_reg",
                    default = 0.0,
                    type = float)
parser.add_argument("--th_type_vol",
                    help = "th_type_reg",
                    default = 0.0,
                    type = float)
parser.add_argument("--alpha_type_vol_l1",
                    help = "alpha_type_vol_l1",
                    default = 0.0,
                    type = float)
parser.add_argument("--alpha_type_vol_l2",
                    help = "alpha_type_vol_l2",
                    default = 0.0,
                    type = float)
parser.add_argument("--marginal_scale",
                    help = "marghinal_scale",
                    default = 1.0,
                    type = float)
parser.add_argument("--avg_pooling",
                    help = "Averaging all hidden states instead of using [CLS].",
                    action = 'store_true')
parser.add_argument("--inv_softplus_temp",
                    help = "Softplus temp.",
                    default = 1.,
                    type = float)
parser.add_argument("--softplus_scale",
                    help = "Softplus scale.",
                    default = 1.,
                    type = float)
parser.add_argument("--proj_layer", help = "Projection type.", default = "linear")
parser.add_argument("--n_proj_layer", help = "n_proj_layer", default = 2, type = int)
parser.add_argument("--use_gumbel_baysian",
                    help = "Make this Ture to use Gumbel box.",
                    default = False,
                    type = bool)
parser.add_argument("--gumbel_beta",
                    help = "Gumbel beta.",
                    default = -1.,
                    type = float)
parser.add_argument("--reduced_type_emb_dim",
                    help = "For baseline. Project down CLS vec.",
                    default = -1,
                    type = int)
parser.add_argument("--alpha_hierarchy_loss",
                    help = "alpha_hierarchy_loss",
                    default = 0.,
                    type = float)
parser.add_argument("--conditional_prob_path",
                    help = "Type conditionals path.",
                    default = "")
parser.add_argument("--marginal_prob_path",
                    help = "Type marginals path.",
                    default = "")
parser.add_argument("--encoder_layer_ids",
                    help = "Encoder's hidden layer IDs that will be used for "
                           "the mentin and context representation. To use the "
                           "last 4 layers of BERT large, it should be "
                           "21 22 23 24, 0-indexed.",
                    nargs = '*',
                    type = int)

# Save / log related
parser.add_argument("--save_period",
                    help = "How often to save.",
                    default = 1000,
                    type = int)
parser.add_argument("--save_best_model",
                    help = "Save the current best model.",
                    default = True,
                    type = bool)
parser.add_argument("--eval_period",
                    help = "How often to run dev.", default = 500, type = int)
parser.add_argument("--log_period",
                    help = "How often to record.",
                    default = 1000,
                    type = int)
parser.add_argument("--eval_after",
                    help = "Start eval after X steps.",
                    default = 10,
                    type = int)
parser.add_argument("--max_num_eval",
                    help = "Stop after evaluating X times.",
                    default = -1,
                    type = int)
parser.add_argument("--load", help = "Load existing model.", action = 'store_true')
parser.add_argument("--reload_model_name",
                    help = "Model name to reload.",
                    default = "")
parser.add_argument("--pretrained_box_path",
                    help = "Pretrained box path.",
                    default = "")
parser.add_argument("--use_wandb",
                    help = "Use Weights & Biases.",
                    default = False,
                    type = bool)
parser.add_argument("--wandb_project_name",
                    help = "Weights & Biases: project name.",
                    default = "")
parser.add_argument("--wandb_username",
                    help = "Weights & Biases: username.",
                    default = "")

"""
Utils
"""

SIGMOID = nn.Sigmoid()


def set_seed(seed: int, n_gpu: int):
    """Set seed for all random number generators."""
    if seed < 0:
        _seed = time.thread_time_ns() % 4294967295
    else:
        _seed = seed
    print("seed: {}".format(_seed))
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(_seed)


def get_data_gen(dataname: str,
                 mode: str,
                 args: argparse.Namespace,
                 tokenizer: object) -> Generator[Dict, None, None]:
    """Returns a data generator."""
    data_path = constant.FILE_ROOT + dataname
    dataset = DatasetLoader(data_path, args, tokenizer)
    if mode == "train":
        data_gen = dataset.get_batch(args.train_batch_size,
                                     args.max_position_embeddings,
                                     args.num_epoch,
                                     eval_data = False)
    else:  # test mode
        data_gen = dataset.get_batch(args.eval_batch_size,
                                     args.max_position_embeddings,
                                     1,  # num epochs is 1.
                                     eval_data = True)
    return data_gen


def get_all_datasets(args: argparse.Namespace, tokenizer: object) -> List[Generator[Dict, None, None]]:
    train_gen_list = []
    if args.mode in ["train"]:
        for train_data_path in args.train_data.split(","):
            train_gen_list.append(
                get_data_gen(train_data_path, "train", args, tokenizer))
    return train_gen_list


def get_datasets(data_lists: List[Tuple[str, str]],
                 args: argparse.Namespace,
                 tokenizer: object) -> List[Generator[Dict, None, None]]:
    data_gen_list = []
    for dataname, mode in data_lists:
        data_gen_list.append(get_data_gen(dataname, mode, args, tokenizer))
    return data_gen_list


def evaluate_data(
        batch_num: int,
        dev_fname: str,
        model: TransformerBoxModelV1,
        id2word_dict: Dict[int, str],
        args: argparse.Namespace,
        device: torch.device
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Computes macro&micro precision, recall, and F1."""
    model.eval()
    dev_gen = get_data_gen(dev_fname, "test", args, model.transformer_tokenizer)
    gold_pred = []
    eval_loss = 0.
    total_ex_count = 0
    for batch in tqdm(dev_gen):
        total_ex_count += len(batch["targets"])
        inputs, targets = to_torch(batch, device)
        loss, output_logits = model(inputs, targets)
        output_index = get_output_index(
            output_logits,
            threshold = args.threshold,
            is_prob = True if args.emb_type == "box" else False)
        gold_pred += get_gold_pred_str(output_index,
                                       batch["targets"].data.cpu().clone(),
                                       id2word_dict)
        eval_loss += loss.clone().item()
    eval_loss /= float(total_ex_count)
    count, pred_count, avg_pred_count, micro_p, micro_r, micro_f1 = micro(gold_pred)
    _, _, _, macro_p, macro_r, macro_f1 = macro(gold_pred)
    accuracy = sum(
        [set(y) == set(yp) for y, yp in gold_pred]) * 1.0 / len(gold_pred)
    eval_loss_str = "Eval loss: {0:.7f} at step {1:d}".format(eval_loss,
                                                              batch_num)
    eval_str = "Eval: {0} {1} {2:.3f} P:{3:.3f} R:{4:.3f} F1:{5:.3f} Ma_P:{" \
               "6:.3f} Ma_R:{7:.3f} Ma_F1:{8:.3f}".format(count,
                                                          pred_count,
                                                          avg_pred_count,
                                                          micro_p,
                                                          micro_r,
                                                          micro_f1,
                                                          macro_p,
                                                          macro_r,
                                                          macro_f1)
    eval_str += "\t Dev EM: {0:.1f}%".format(accuracy * 100)
    print("==>  EVAL: seen " + repr(total_ex_count) + " examples.")
    print(eval_loss_str)
    print(gold_pred[:3])
    print("==> " + eval_str)
    model.train()
    dev_gen = None  # Delete a data generator.
    return eval_loss, macro_f1, macro_p, macro_r, micro_f1, micro_p, micro_r, \
           accuracy


def f1(p: float, r: float) -> float:
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def macro(
        true_and_prediction: List[Tuple[List[str], List[str]]]
) -> Tuple[int, int, int, float, float, float]:
    """Computes macro precision, recall, and F1."""
    num_examples = len(true_and_prediction)
    p = 0.
    r = 0.
    pred_example_count = 0
    pred_label_count = 0.
    gold_label_count = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if predicted_labels:
            pred_example_count += 1
            pred_label_count += len(predicted_labels)
            per_p = len(set(predicted_labels).intersection(set(true_labels))) / \
                    float(len(predicted_labels))
            p += per_p
        if len(true_labels):
            gold_label_count += 1
            per_r = len(set(predicted_labels).intersection(set(true_labels))) / \
                    float(len(true_labels))
            r += per_r
    if pred_example_count == 0 or gold_label_count == 0:
        return num_examples, 0, 0, 0., 0., 0.
    precision = p / float(pred_example_count)
    recall = r / gold_label_count
    avg_elem_per_pred = pred_label_count / float(pred_example_count)
    return num_examples, pred_example_count, avg_elem_per_pred, precision, \
           recall, f1(precision, recall)


def micro(
        true_and_prediction: List[Tuple[List[str], List[str]]]
) -> Tuple[int, int, int, float, float, float]:
    """Computes micro precision, recall, and F1."""
    num_examples = len(true_and_prediction)
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    pred_example_count = 0
    for true_labels, predicted_labels in true_and_prediction:
        if predicted_labels:
            pred_example_count += 1
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(
            set(predicted_labels).intersection(set(true_labels)))
    if pred_example_count == 0 or num_predicted_labels == 0 \
            or num_true_labels == 0:
        return num_examples, 0, 0, 0., 0., 0.
    precision = num_correct_labels / num_predicted_labels
    recall = num_correct_labels / num_true_labels
    avg_elem_per_pred = num_predicted_labels / float(pred_example_count)
    return num_examples, pred_example_count, avg_elem_per_pred, precision, \
           recall, f1(precision, recall)


def load_model(reload_model_name: str,
               save_dir: str,
               model_id: str,
               model: TransformerBoxModelV1,
               optimizer_enc: Optional[torch.optim.Optimizer] = None,
               optimizer_cls: Optional[torch.optim.Optimizer] = None,
               scheduler_enc: Optional[object] = None,
               scheduler_cls: Optional[object] = None):
    if reload_model_name:
        model_file_name = "{0:s}/{1:s}.pt".format(save_dir, reload_model_name)
    else:
        model_file_name = "{0:s}/{1:s}.pt".format(save_dir, model_id)
    checkpoint = torch.load(model_file_name)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print("Loading model from ... {0:s}".format(model_file_name))


def get_output_index(outputs: torch.Tensor,
                     threshold: float = 0.5,
                     is_prob: bool = False) -> List[List[int]]:
    """Given outputs from the decoder, generates prediction index."""
    pred_idx = []
    if is_prob:
        outputs = outputs.data.cpu().clone()
    else:
        outputs = SIGMOID(outputs).data.cpu().clone()
    for single_dist in outputs:
        single_dist = single_dist.numpy()
        arg_max_ind = np.argmax(single_dist)
        pred_id = [arg_max_ind]
        pred_id.extend(
            [i for i in range(len(single_dist))
             if single_dist[i] > threshold and i != arg_max_ind])
        pred_idx.append(pred_id)
    return pred_idx


def get_gold_pred_str(
        pred_idx: List[List[int]],
        gold: List[List[int]],
        id2word_dict: Dict[int, str]
) -> List[Tuple[List[str], List[str]]]:
    """
    Given predicted ids and gold ids, generate a list of (gold, pred) pairs of
    length batch_size.
    """
    gold_strs = []
    for gold_i in gold:
        gold_strs.append([id2word_dict[i] for i in range(len(gold_i))
                          if gold_i[i] == 1])
    pred_strs = []
    for pred_idx1 in pred_idx:
        pred_strs.append([(id2word_dict[ind]) for ind in pred_idx1])
    else:
        return list(zip(gold_strs, pred_strs))


def get_eval_string(true_prediction: List[Tuple[List[str], List[str]]]) -> str:
    """Returns an eval results string."""
    count, pred_count, avg_pred_count, p, r, f1 = micro(true_prediction)
    _, _, _, ma_p, ma_r, ma_f1 = macro(true_prediction)
    output_str = "Eval: {0} {1} {2:.3f} P:{3:.3f} R:{4:.3f} F1:{5:.3f} Ma_P:{" \
                 "6:.3f} Ma_R:{7:.3f} Ma_F1:{8:.3f}".format(count,
                                                            pred_count,
                                                            avg_pred_count,
                                                            p,
                                                            r,
                                                            f1,
                                                            ma_p,
                                                            ma_r,
                                                            ma_f1)
    accuracy = sum([set(y) == set(yp) for y, yp in true_prediction]) * 1.0 \
               / len(true_prediction)
    output_str += "\t Dev EM: {0:.1f}%".format(accuracy * 100)
    return output_str


def _test(args: argparse.Namespace,
          model: TransformerBoxModelV1,
          device: torch.device):
    print("==> Start eval...")
    assert args.load
    save_output_to = os.path.join(constant.BASE_PATH, "outputs", args.model_id)
    if not os.path.exists(save_output_to):
        print("==> Create {}".format(save_output_to))
        os.makedirs(save_output_to, exist_ok = False)
    test_fname = args.eval_data
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    print("==> Loading data generator... ")
    data_gens = get_datasets([(test_fname, "test")],
                             args,
                             model.transformer_tokenizer)
    print("==> Loading ID -> TYPE mapping... ")
    _word2id = constant.load_vocab_dict(constant.TYPE_FILES[args.goal])
    id2word_dict = {v: k for k, v in _word2id.items()}
    model.eval()
    load_model(args.reload_model_name,
               constant.EXP_ROOT,
               args.model_id, model)
    for name, dataset in [(test_fname, data_gens[0])]:
        print("Processing... " + name)
        total_gold_pred = []
        total_annot_ids = []
        total_probs = []
        total_ys = []
        for batch_num, batch in enumerate(dataset):
            inputs, targets = to_torch(batch, device)
            annot_ids = batch.pop("ex_ids")
            output_logits = model(inputs)
            output_index = get_output_index(
                output_logits,
                threshold = args.threshold,
                is_prob = True if args.emb_type == "box" else False)
            output_prob = output_logits.data.cpu().clone().numpy()
            y = batch["targets"].data.cpu().clone()
            gold_pred = get_gold_pred_str(output_index, y, id2word_dict)
            total_probs.extend(output_prob)
            total_ys.extend(y)
            total_gold_pred.extend(gold_pred)
            total_annot_ids.extend(annot_ids)
        pickle.dump(
            {"gold_id_array": total_ys, "pred_dist": total_probs},
            open("{0:s}/pred_dist.pkl".format(save_output_to), "wb"))
        print(len(total_annot_ids), len(total_gold_pred))
        with open("{0:s}/pred_labels.json".format(save_output_to), "w") as f_out:
            output_dict = {}
            counter = 0
            for a_id, (gold, pred) in zip(total_annot_ids, total_gold_pred):
                output_dict[a_id] = {"gold": gold, "pred": pred}
                counter += 1
            json.dump(output_dict, f_out)
        eval_str = get_eval_string(total_gold_pred)
        print(eval_str)


def main():
    args = parser.parse_args()
    # Lower text for BERT uncased models
    args.do_lower = True if "uncased" in args.model_type else False
    device = torch.device("cuda")
    args.n_gpu = 1
    args.device = device
    set_seed(args.seed, args.n_gpu)
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    model = TransformerBoxModelV1(args, constant.ANSWER_NUM_DICT[args.goal])
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)
    args.max_position_embeddings = \
        model.transformer_config.max_position_embeddings
    _test(args, model, device)


if __name__ == "__main__":
    main()
