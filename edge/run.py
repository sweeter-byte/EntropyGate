#!/usr/bin/env python3
"""EDGE: Entropy-Driven Gated Decoding — Main execution script.

Usage:
    python edge/run.py --experiment_name my_exp --run_chair_benchmark --load_in_4bit
    python edge/run.py --experiment_name my_exp --run_pope_benchmark --load_in_4bit
    python edge/run.py --experiment_name my_exp --run_mme_benchmark --load_in_4bit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.generation_config import EdgeGenerationConfig
from edge.constants import (
    DEFAULT_ALPHA_BASE_VIS, DEFAULT_ALPHA_MIN_VIS,
    DEFAULT_ETA_VIS, DEFAULT_TAU_GATE, DEFAULT_GAMMA_DECAY,
    DEFAULT_BETA_CUTOFF, DEFAULT_THETA_SAFE,
    DEFAULT_AGGREGATE_LAYER_FAST_V, DEFAULT_MINUMUM_FAST_V_TOKENS,
    DEFAULT_AGGREGATE_LAYER_TEXT_MASK, DEFAULT_MINIMUM_TEXT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K,
    get_image_token_id,
)
from edge.benchmark.chair import ChairBenchmarkDataset
from edge.benchmark.amber import AmberBenchmarkDataset
from edge.benchmark.pope import PopeBenchmarkDataset
from edge.benchmark.evaluators.mme_utils import parse_pred_ans, eval_type_dict
from edge.utils.reproducibility import set_reproducibility

from collections import defaultdict
import logging
import torch
import gc
import json
import re
import argparse
import numpy as np
from tqdm.auto import tqdm
from accelerate import PartialState
from accelerate.utils import gather_object
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers import logging as hf_logging
from datasets import load_dataset
hf_logging.set_verbosity_error()

distributed_state = PartialState()


def _model_slug(model_name: str) -> str:
    return "--".join(part for part in model_name.split("/") if part)


def args_parser():
    parser = argparse.ArgumentParser(description="EDGE: Entropy-Driven Gated Decoding")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--load_in_8bit", action='store_true', default=False)
    parser.add_argument("--load_in_4bit", action='store_true', default=False)

    # Generation config
    parser.add_argument("--do_sample", action='store_true', default=False)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    # Attention mask layers (inherited from CRoPS)
    parser.add_argument("--aggregate_layer_fast_v", type=int, default=DEFAULT_AGGREGATE_LAYER_FAST_V)
    parser.add_argument("--minumum_fast_v_tokens", type=int, default=DEFAULT_MINUMUM_FAST_V_TOKENS)
    parser.add_argument("--aggregate_layer_text_mask", type=int, default=DEFAULT_AGGREGATE_LAYER_TEXT_MASK)
    parser.add_argument("--minimum_text_tokens", type=int, default=DEFAULT_MINIMUM_TEXT_TOKENS)

    # EDGE core parameters
    parser.add_argument("--alpha_base_vis", type=float, default=DEFAULT_ALPHA_BASE_VIS,
                        help="Visual entropy gate upper bound (default: 1.5)")
    parser.add_argument("--alpha_min_vis", type=float, default=DEFAULT_ALPHA_MIN_VIS,
                        help="Visual entropy gate lower bound (default: 0.5)")
    parser.add_argument("--eta_vis", type=float, default=DEFAULT_ETA_VIS,
                        help="Entropy threshold for visual gating (default: 0.1)")
    parser.add_argument("--tau_gate", type=float, default=DEFAULT_TAU_GATE,
                        help="Gate temperature / sigmoid sharpness (default: 0.05)")
    parser.add_argument("--gamma_decay", type=float, default=DEFAULT_GAMMA_DECAY,
                        help="Time decay coefficient for lang prior (default: 0.01)")
    parser.add_argument("--beta_cutoff", type=float, default=DEFAULT_BETA_CUTOFF,
                        help="Plausibility cutoff threshold (default: 0.1)")
    parser.add_argument("--theta_safe", type=float, default=DEFAULT_THETA_SAFE,
                        help="Safety skip threshold (default: 0.99)")

    # Evaluation config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, required=True)

    # Benchmarks
    parser.add_argument("--run_chair_benchmark", action='store_true', default=False)
    parser.add_argument("--coco_path", type=str, default='dataset/annotations')
    parser.add_argument("--coco_file", type=str, default='instances_val2014.json')
    parser.add_argument("--coco_base_image_path", type=str, default='dataset/val2014')
    parser.add_argument("--chair_test_size", type=int, default=500)

    parser.add_argument("--run_amber_benchmark", action='store_true', default=False)
    parser.add_argument("--amber_query_file", type=str, default='/data1/ranmaoyin/dataset/amber/data/query/query_generative.json')
    parser.add_argument("--amber_image_dir", type=str, default='/data1/ranmaoyin/dataset/amber/images')
    parser.add_argument("--amber_official_repo_path", type=str, default='/data1/ranmaoyin/dataset/amber/official_repo')
    parser.add_argument("--amber_evaluation_type", type=str, default='g', choices=['a', 'g', 'd', 'de', 'da', 'dr'])

    parser.add_argument("--run_pope_benchmark", action='store_true', default=False)
    parser.add_argument("--pope_path", type=str, default='/data1/ranmaoyin/dataset/pope')
    parser.add_argument("--pope_coco_image_dir", type=str, default='/data1/ranmaoyin/dataset/coco2014/val2014')
    parser.add_argument("--pope_splits", type=str, nargs='+', default=['random', 'popular', 'adversarial'],
                        choices=['random', 'popular', 'adversarial'])

    parser.add_argument("--run_mme_benchmark", action='store_true', default=False)

    return parser.parse_args()


def setup_logging(experiment_name, log_level="INFO"):
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")

    eg_logger = logging.getLogger("edge")
    eg_logger.setLevel(getattr(logging, log_level))
    eg_logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    eg_logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level))
    ch.setFormatter(fmt)
    eg_logger.addHandler(ch)

    return eg_logger, log_file


def make_generation_config(args, image_tokens, input_ids_lang_prior):
    return EdgeGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        do_sample=args.do_sample,
        use_cache=True,
        key_position={
            "image_start": image_tokens[0],
            "image_end": image_tokens[-1],
        },
        input_ids_lang_prior=input_ids_lang_prior,
        aggregate_layer_fast_v=args.aggregate_layer_fast_v,
        minumum_fast_v_tokens=args.minumum_fast_v_tokens,
        aggregate_layer_text_mask=args.aggregate_layer_text_mask,
        minimum_text_tokens=args.minimum_text_tokens,
        alpha_base_vis=args.alpha_base_vis,
        alpha_min_vis=args.alpha_min_vis,
        eta_vis=args.eta_vis,
        tau_gate=args.tau_gate,
        gamma_decay=args.gamma_decay,
        beta_cutoff=args.beta_cutoff,
        theta_safe=args.theta_safe,
    )


# ---- Model format helpers ----

_USE_QWEN_FORMAT = False

_LLAVA_SYSTEM_CONTENT = [{"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}]


def _is_qwen_model(model_name: str) -> bool:
    _QWEN_KEYWORDS = ("qwen", "r1-onevision", "vision-r1", "vl-rethinker", "vl-cogito", "openvlthinker")
    name_lower = model_name.lower()
    return any(kw in name_lower for kw in _QWEN_KEYWORDS)


def _strip_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _decode_output(processor, output_ids, input_len: int) -> str:
    text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    if _USE_QWEN_FORMAT:
        text = _strip_think_tags(text)
    return text


def _build_lang_prior_inputs(processor, text, device):
    conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    if not _USE_QWEN_FORMAT:
        conversation.insert(0, {"role": "system", "content": _LLAVA_SYSTEM_CONTENT})
    out = processor.apply_chat_template(
        conversation, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device, torch.bfloat16)
    return out["input_ids"]


def _build_full_inputs(processor, image, text, device):
    conversation = [{"role": "user", "content": [
        {"type": "image", "url": image},
        {"type": "text", "text": text},
    ]}]
    if not _USE_QWEN_FORMAT:
        conversation.insert(0, {"role": "system", "content": _LLAVA_SYSTEM_CONTENT})
    return processor.apply_chat_template(
        conversation, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device, torch.bfloat16)


# ---- Benchmark runners ----

def run_chair_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", _model_slug(args.model_name), "EDGE", args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)

    chair_benchmark = ChairBenchmarkDataset(
        coco_path=args.coco_path, coco_file=args.coco_file,
        base_image_path=args.coco_base_image_path, chair_test_size=args.chair_test_size,
    )

    with distributed_state.local_main_process_first():
        test_dataset = chair_benchmark.get_test_dataset()

    prompt_text = "Please describe this image in detail"
    input_ids_lang_prior = _build_lang_prior_inputs(processor, prompt_text, distributed_state.device)
    image_token_ids = get_image_token_id(args.model_name)

    with distributed_state.split_between_processes(test_dataset) as process_test_dataset:
        generations = []
        for test_image in tqdm(process_test_dataset, desc=f"EDGE CHAIR [{distributed_state.process_index}]"):
            inputs = _build_full_inputs(processor, test_image["image_path"], prompt_text, distributed_state.device)
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]
            generation_config = make_generation_config(args, image_tokens, input_ids_lang_prior)

            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = _decode_output(processor, output_ids, len(inputs["input_ids"][0]))

            try:
                del inputs["input_ids"]
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            generations.append({
                chair_benchmark.image_id_key: test_image["image_id"],
                chair_benchmark.caption_key: output_text,
            })

    generations = gather_object(generations)
    if distributed_state.is_main_process:
        generations_path = os.path.join(experiment_name, "chair_generations.jsonl")
        chair_benchmark.dump_generations(generations, generations_path)
        chair_benchmark.evaluate(generations_path, dump_results=True)


def run_pope_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", _model_slug(args.model_name), "EDGE", "POPE", args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)

    pope_benchmark = PopeBenchmarkDataset(
        pope_path=args.pope_path, coco_image_dir=args.pope_coco_image_dir, pope_splits=args.pope_splits,
    )

    with distributed_state.local_main_process_first():
        test_dataset = pope_benchmark.get_test_dataset()

    image_token_ids = get_image_token_id(args.model_name)
    lang_prior_cache = {}

    with distributed_state.split_between_processes(test_dataset) as process_test_dataset:
        generations = []
        for sample in tqdm(process_test_dataset, desc=f"EDGE POPE [{distributed_state.process_index}]"):
            prompt_text = sample["prompt"]
            if prompt_text not in lang_prior_cache:
                lang_prior_cache[prompt_text] = _build_lang_prior_inputs(processor, prompt_text, distributed_state.device)
            input_ids_lang_prior = lang_prior_cache[prompt_text]

            inputs = _build_full_inputs(processor, sample["image_path"], prompt_text, distributed_state.device)
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]
            generation_config = make_generation_config(args, image_tokens, input_ids_lang_prior)

            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = _decode_output(processor, output_ids, len(inputs["input_ids"][0]))

            del output_ids, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            generations.append({
                "question_id": sample["question_id"],
                "split": sample["split"],
                "label": sample["label"],
                "response": output_text,
            })

    generations = gather_object(generations)
    if distributed_state.is_main_process:
        generations_path = os.path.join(experiment_name, "pope_generations.jsonl")
        pope_benchmark.dump_generations(generations, generations_path)
        pope_benchmark.evaluate(generations_path, dump_results=True)


def run_amber_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", _model_slug(args.model_name), "EDGE", "AMBER", args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)

    amber_benchmark = AmberBenchmarkDataset(
        query_file=args.amber_query_file, image_dir=args.amber_image_dir,
        official_repo_path=args.amber_official_repo_path,
    )

    with distributed_state.local_main_process_first():
        test_dataset = amber_benchmark.get_test_dataset()

    image_token_ids = get_image_token_id(args.model_name)
    lang_prior_cache = {}

    with distributed_state.split_between_processes(test_dataset) as process_test_dataset:
        generations = []
        for sample in tqdm(process_test_dataset, desc=f"EDGE AMBER [{distributed_state.process_index}]"):
            prompt_text = sample["prompt"]
            if prompt_text not in lang_prior_cache:
                lang_prior_cache[prompt_text] = _build_lang_prior_inputs(processor, prompt_text, distributed_state.device)
            input_ids_lang_prior = lang_prior_cache[prompt_text]

            inputs = _build_full_inputs(processor, sample["image_path"], prompt_text, distributed_state.device)
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]
            generation_config = make_generation_config(args, image_tokens, input_ids_lang_prior)

            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = _decode_output(processor, output_ids, len(inputs["input_ids"][0]))

            del output_ids, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            generations.append({"id": sample["id"], "response": output_text})

    generations = gather_object(generations)
    if distributed_state.is_main_process:
        generations_path = os.path.join(experiment_name, "amber_generations.json")
        evaluation_output_path = os.path.join(experiment_name, "amber_evaluation.txt")
        amber_benchmark.dump_generations(generations, generations_path)
        evaluation_output = amber_benchmark.evaluate(
            generations_path, evaluation_type=args.amber_evaluation_type,
            dump_results=True, evaluation_output_path=evaluation_output_path,
        )
        print("AMBER evaluation complete.")
        print(evaluation_output)


def run_mme_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", _model_slug(args.model_name), "EDGE", "MME", args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)

    mme_dataset = load_dataset("darkyarding/MME")["test"]
    data_list = list(mme_dataset)
    image_token_ids = get_image_token_id(args.model_name)

    with distributed_state.split_between_processes(data_list) as process_data_list:
        results = []
        for sample in tqdm(process_data_list, total=len(process_data_list), desc=f"EDGE MME [{distributed_state.process_index}]"):
            question = sample["question"]
            gt_ans = sample["answer"].lower()

            input_ids_lang_prior = _build_lang_prior_inputs(processor, question, distributed_state.device)
            inputs = _build_full_inputs(processor, sample["image"], question, distributed_state.device)
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]
            generation_config = make_generation_config(args, image_tokens, input_ids_lang_prior)

            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = _decode_output(processor, output_ids, len(inputs["input_ids"][0]))
            pred_ans = parse_pred_ans(output_text)

            results.append({
                "question_id": sample["question_id"],
                "category": sample["category"],
                "pred_ans": pred_ans,
                "gt_ans": gt_ans,
            })

            del output_ids, inputs, input_ids_lang_prior
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    results = gather_object(results)
    if distributed_state.is_main_process:
        question_pairs = defaultdict(list)
        for res in results:
            question_pairs[res["question_id"]].append(res)

        category2score = defaultdict(list)
        for question_id, samples in question_pairs.items():
            assert len(samples) == 2, f"Question ID {question_id} does not have a pair!"
            score_1 = 1.0 if samples[0]["pred_ans"] == samples[0]["gt_ans"] else 0.0
            score_2 = 1.0 if samples[1]["pred_ans"] == samples[1]["gt_ans"] else 0.0
            acc = (score_1 + score_2) / 2 * 100.0
            acc_plus = 100.0 if score_1 == 1.0 and score_2 == 1.0 else 0.0
            category2score[samples[0]["category"]].append(acc + acc_plus)

        category2avg_score = {cat: sum(s) / len(s) for cat, s in category2score.items()}
        perception_score = sum(category2avg_score.get(cat, 0) for cat in eval_type_dict["Perception"])
        cognition_score = sum(category2avg_score.get(cat, 0) for cat in eval_type_dict["Cognition"])

        with open(os.path.join(experiment_name, 'mme_results.txt'), "a") as f:
            f.write(f"{args}\n")
            f.write("=========== Perception ===========\n")
            f.write(f"total score: {perception_score:.2f}\n\n")
            for category in eval_type_dict["Perception"]:
                f.write(f"\t {category}  score: {category2avg_score.get(category, 0):.2f}\n")
            f.write("\n=========== Cognition ===========\n")
            f.write(f"total score: {cognition_score:.2f}\n\n")
            for category in eval_type_dict["Cognition"]:
                f.write(f"\t {category}  score: {category2avg_score.get(category, 0):.2f}\n")

        print("MME evaluation complete. Results saved.")


def main():
    args = args_parser()
    set_reproducibility(args.seed)

    # Apply EDGE monkey-patches
    from edge.method import patch_everything
    patch_everything()

    # Setup logging
    eg_logger, log_file = setup_logging(args.experiment_name)
    eg_logger.info(f"EDGE experiment: {args.experiment_name}")
    eg_logger.info(f"Log file: {log_file}")
    eg_logger.info(f"Model: {args.model_name}")
    eg_logger.info(
        f"Hyperparams: alpha_vis=[{args.alpha_min_vis},{args.alpha_base_vis}] "
        f"eta_vis={args.eta_vis} tau={args.tau_gate} "
        f"gamma={args.gamma_decay} beta_cutoff={args.beta_cutoff} theta_safe={args.theta_safe}"
    )
    eg_logger.info(f"Full args: {vars(args)}")

    if args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": distributed_state.device},
        quantization_config=bnb_config, attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    global _USE_QWEN_FORMAT
    _USE_QWEN_FORMAT = _is_qwen_model(args.model_name)
    if _USE_QWEN_FORMAT:
        print(f"Detected Qwen2.5-VL series model: {args.model_name}")

    if args.run_chair_benchmark:
        run_chair_benchmark(model, processor, args)
    if args.run_amber_benchmark:
        run_amber_benchmark(model, processor, args)
    if args.run_mme_benchmark:
        run_mme_benchmark(model, processor, args)
    if args.run_pope_benchmark:
        run_pope_benchmark(model, processor, args)


if __name__ == "__main__":
    main()
