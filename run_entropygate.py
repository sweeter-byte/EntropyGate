import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from methods.generation_configs.entropygate_generation_config import GenerationConfigEntropyGate
from methods.generation_configs.contrastive_generation_config import GenerationConfigContrastive
from methods.generation_configs.latent_generation_config import GenerationConfigLatent

from constants.vcd_constants import (
    DEFAULT_VCD_ALPHA,
    DEFAULT_VCD_BETA,
    DEFAULT_VCD_NOISE_STEP,
)

from constants.entropygate_constants import (
    DEFAULT_ALPHA_BASE_VIS,
    DEFAULT_ALPHA_BASE_TXT,
    DEFAULT_ALPHA_MIN_VIS,
    DEFAULT_ALPHA_MIN_TXT,
    DEFAULT_ETA_VIS,
    DEFAULT_ETA_TXT,
    DEFAULT_TAU_GATE,
    DEFAULT_GAMMA_DECAY,
    DEFAULT_BETA_BASE,
    DEFAULT_BETA_RANGE,
    DEFAULT_THETA_SAFE,
    DEFAULT_EG_SCHEME,
    DEFAULT_BETA_CUTOFF_FIXED,
    DEFAULT_THETA_SAFE_ALIGNED,
    DEFAULT_TIME_DECAY_MODE,
    DEFAULT_ALPHA_TIME_TXT,
    DEFAULT_ADAPTIVE_ETA,
    DEFAULT_ETA_EMA_MOMENTUM,
    DEFAULT_ETA_VIS_OFFSET,
    DEFAULT_ETA_TXT_OFFSET,
    DEFAULT_SOFT_SUPPRESS,
    DEFAULT_SOFT_SUPPRESS_K,
)

from constants.crops_constants import (
    DEFAULT_AGGREGATE_LAYER_FAST_V,
    DEFAULT_MINUMUM_FAST_V_TOKENS,
    DEFAULT_AGGREGATE_LAYER_TEXT_MASK,
    DEFAULT_MINIMUM_TEXT_TOKENS,
    DEFAULT_LAMBDA_LANG_PRIOR,
    DEFAULT_ALPHA_STAT_BIAS,
    DEFAULT_BETA_CUTOFF,
    DEFAULT_MAX_THRESHOLD_PLAUSIBILITY_CONSTRAINT,
)

from constants.latent_constants import (
    DEFAULT_HSC_ALPHA_BASE,
    DEFAULT_HSC_ALPHA_MIN,
    DEFAULT_HSC_ETA,
    DEFAULT_HSC_TAU,
    DEFAULT_LEG_HIDDEN_LAYER,
    DEFAULT_LLH_HIDDEN_ALPHA_BASE,
    DEFAULT_LLH_HIDDEN_ALPHA_MIN,
    DEFAULT_LLH_HIDDEN_ETA,
    DEFAULT_LLH_HIDDEN_TAU,
    DEFAULT_LATENT_METHOD,
)

from constants.image_token_constants import get_image_token_id

from constants.default_generation_constants import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
)

from benchmark.chair_benchmark import ChairBenchmarkDataset
from benchmark.amber_benchmark import AmberBenchmarkDataset
from benchmark.evaluators.mme.utils import parse_pred_ans, eval_type_dict
try:
    from benchmark.mmmu_utils import CAT_SHORT2LONG, construct_prompt, process_single_sample, evaluate_mmmu
    _HAS_MMMU = True
except ImportError:
    _HAS_MMMU = False
from utils.reproducibility_util import set_reproducibility

from collections import defaultdict

import os
import logging
import torch
import gc
import json
import argparse
import numpy as np
from tqdm.auto import tqdm
from accelerate import PartialState
from accelerate.utils import gather_object
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers import logging as hf_logging
from datasets import load_dataset, concatenate_datasets
hf_logging.set_verbosity_error()

distributed_state = PartialState()


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="entropygate",
                        choices=["vanilla", "crops", "entropygate", "vcd", "vcd_eg", "latent"],
                        help="Decoding method: vanilla, crops, entropygate, vcd, vcd_eg, latent (HSC/LEG/LLH)")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--load_in_8bit", action='store_true', default=False)
    parser.add_argument("--load_in_4bit", action='store_true', default=False)

    # Generation config
    parser.add_argument("--do_sample", action='store_true', default=False)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    # CRoPS shared config (attention mask layers)
    parser.add_argument("--aggregate_layer_fast_v", type=int, default=DEFAULT_AGGREGATE_LAYER_FAST_V)
    parser.add_argument("--minumum_fast_v_tokens", type=int, default=DEFAULT_MINUMUM_FAST_V_TOKENS)
    parser.add_argument("--aggregate_layer_text_mask", type=int, default=DEFAULT_AGGREGATE_LAYER_TEXT_MASK)
    parser.add_argument("--minimum_text_tokens", type=int, default=DEFAULT_MINIMUM_TEXT_TOKENS)

    # EntropyGate-specific config
    parser.add_argument("--alpha_base_vis", type=float, default=DEFAULT_ALPHA_BASE_VIS)
    parser.add_argument("--alpha_base_txt", type=float, default=DEFAULT_ALPHA_BASE_TXT)
    parser.add_argument("--alpha_min_vis", type=float, default=DEFAULT_ALPHA_MIN_VIS)
    parser.add_argument("--alpha_min_txt", type=float, default=DEFAULT_ALPHA_MIN_TXT)
    parser.add_argument("--eta_vis", type=float, default=DEFAULT_ETA_VIS)
    parser.add_argument("--eta_txt", type=float, default=DEFAULT_ETA_TXT)
    parser.add_argument("--tau_gate", type=float, default=DEFAULT_TAU_GATE)
    parser.add_argument("--gamma_decay", type=float, default=DEFAULT_GAMMA_DECAY)
    parser.add_argument("--beta_base", type=float, default=DEFAULT_BETA_BASE)
    parser.add_argument("--beta_range", type=float, default=DEFAULT_BETA_RANGE)
    parser.add_argument("--theta_safe", type=float, default=DEFAULT_THETA_SAFE)
    parser.add_argument("--eg_scheme", type=str, default=DEFAULT_EG_SCHEME,
                        choices=["flat", "nested", "acd", "nested_aligned"],
                        help="Contrastive formula scheme: flat (D/D+A), nested (E), acd (F), nested_aligned (G)")
    parser.add_argument("--beta_cutoff_fixed", type=float, default=DEFAULT_BETA_CUTOFF_FIXED)
    parser.add_argument("--theta_safe_aligned", type=float, default=DEFAULT_THETA_SAFE_ALIGNED)

    # Direction 2: time decay mode
    parser.add_argument("--time_decay_mode", type=str, default=DEFAULT_TIME_DECAY_MODE,
                        choices=["multiply", "additive"])
    parser.add_argument("--alpha_time_txt", type=float, default=DEFAULT_ALPHA_TIME_TXT)
    # Direction 3: adaptive eta
    parser.add_argument("--adaptive_eta", action='store_true', default=DEFAULT_ADAPTIVE_ETA)
    parser.add_argument("--eta_ema_momentum", type=float, default=DEFAULT_ETA_EMA_MOMENTUM)
    parser.add_argument("--eta_vis_offset", type=float, default=DEFAULT_ETA_VIS_OFFSET)
    parser.add_argument("--eta_txt_offset", type=float, default=DEFAULT_ETA_TXT_OFFSET)
    # Direction 4: soft suppress
    parser.add_argument("--soft_suppress", action='store_true', default=DEFAULT_SOFT_SUPPRESS)
    parser.add_argument("--soft_suppress_k", type=float, default=DEFAULT_SOFT_SUPPRESS_K)

    # CRoPS-specific config
    parser.add_argument("--lambda_lang_prior", type=float, default=DEFAULT_LAMBDA_LANG_PRIOR)
    parser.add_argument("--alpha_stat_bias", type=float, default=DEFAULT_ALPHA_STAT_BIAS)
    parser.add_argument("--beta_cutoff", type=float, default=DEFAULT_BETA_CUTOFF)
    parser.add_argument("--max_threshold_plausibility_constraint", type=float, default=DEFAULT_MAX_THRESHOLD_PLAUSIBILITY_CONSTRAINT)

    # VCD-specific config
    parser.add_argument("--vcd_alpha", type=float, default=DEFAULT_VCD_ALPHA)
    parser.add_argument("--vcd_beta", type=float, default=DEFAULT_VCD_BETA)
    parser.add_argument("--vcd_noise_step", type=int, default=DEFAULT_VCD_NOISE_STEP)
    # VCD+EntropyGate config (reuses eta_vis, tau_gate from EntropyGate)
    parser.add_argument("--vcd_eg_alpha_min", type=float, default=0.5)
    parser.add_argument("--vcd_eg_alpha_max", type=float, default=1.5)

    # Latent-space method config (HSC/LEG/LLH)
    parser.add_argument("--latent_method", type=str, default=DEFAULT_LATENT_METHOD,
                        choices=["hsc", "leg", "llh"],
                        help="Latent method: hsc (Hidden State Contrastive), leg (Latent Entropy Gate), llh (Latent-Logit Hybrid)")
    # HSC params
    parser.add_argument("--hsc_alpha_base", type=float, default=DEFAULT_HSC_ALPHA_BASE)
    parser.add_argument("--hsc_alpha_min", type=float, default=DEFAULT_HSC_ALPHA_MIN)
    parser.add_argument("--hsc_eta", type=float, default=DEFAULT_HSC_ETA)
    parser.add_argument("--hsc_tau", type=float, default=DEFAULT_HSC_TAU)
    # LEG params
    parser.add_argument("--leg_hidden_layer", type=int, default=DEFAULT_LEG_HIDDEN_LAYER,
                        help="Index into outputs.hidden_states for LEG entropy. "
                             "-16 = roughly mid-layer for 32-layer models.")
    # LLH params
    parser.add_argument("--llh_hidden_alpha_base", type=float, default=DEFAULT_LLH_HIDDEN_ALPHA_BASE)
    parser.add_argument("--llh_hidden_alpha_min", type=float, default=DEFAULT_LLH_HIDDEN_ALPHA_MIN)
    parser.add_argument("--llh_hidden_eta", type=float, default=DEFAULT_LLH_HIDDEN_ETA)
    parser.add_argument("--llh_hidden_tau", type=float, default=DEFAULT_LLH_HIDDEN_TAU)

    # Evaluation config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, required=True)

    # Benchmarks
    parser.add_argument("--run_mmmu_benchmark", action='store_true', default=False)
    parser.add_argument("--mmmu_answer_file_path", type=str, default='benchmark/evaluators/mmmu/answer_dict_val.json')
    parser.add_argument("--run_mathvista_benchmark", action='store_true', default=False)
    parser.add_argument("--run_mme_benchmark", action='store_true', default=False)
    parser.add_argument("--run_chair_benchmark", action='store_true', default=False)
    parser.add_argument("--run_amber_benchmark", action='store_true', default=False)
    parser.add_argument("--coco_path", type=str, default='dataset/annotations')
    parser.add_argument("--coco_file", type=str, default='instances_val2014.json')
    parser.add_argument("--coco_base_image_path", type=str, default='dataset/val2014')
    parser.add_argument("--chair_test_size", type=int, default=500)
    parser.add_argument("--amber_query_file", type=str, default='/data1/ranmaoyin/dataset/amber/data/query/query_generative.json')
    parser.add_argument("--amber_image_dir", type=str, default='/data1/ranmaoyin/dataset/amber/images')
    parser.add_argument("--amber_official_repo_path", type=str, default='/data1/ranmaoyin/dataset/amber/official_repo')
    parser.add_argument("--amber_evaluation_type", type=str, default='g',
                        choices=['a', 'g', 'd', 'de', 'da', 'dr'])

    return parser.parse_args()


def setup_logging(experiment_name, log_level="INFO"):
    """Configure logging to both console and file."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")

    # Root logger for entropygate
    eg_logger = logging.getLogger("entropygate")
    eg_logger.setLevel(getattr(logging, log_level))
    eg_logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler — always DEBUG level to capture per-step details
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    eg_logger.addHandler(fh)

    # Console handler — follows user-specified level
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level))
    ch.setFormatter(fmt)
    eg_logger.addHandler(ch)

    return eg_logger, log_file


def make_generation_config(args, image_tokens, input_ids_lang_prior):
    """Build generation config from CLI args based on selected method."""
    common_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        do_sample=args.do_sample,
        use_cache=True,
    )

    if args.method == "vanilla":
        # Plain sampling — use the base GenerationConfig from transformers
        from transformers import GenerationConfig
        return GenerationConfig(**common_kwargs)

    if args.method in ("vcd", "vcd_eg"):
        from methods.generation_configs.vcd_generation_config import GenerationConfigVCD
        cfg = GenerationConfigVCD(
            **common_kwargs,
            vcd_alpha=args.vcd_alpha,
            vcd_beta=args.vcd_beta,
            vcd_noise_step=args.vcd_noise_step,
        )
        # Attach EntropyGate params for vcd_eg mode
        cfg.vcd_entropy_gate = (args.method == "vcd_eg")
        cfg.eg_alpha_min = args.vcd_eg_alpha_min
        cfg.eg_alpha_max = args.vcd_eg_alpha_max
        cfg.eg_eta = args.eta_vis
        cfg.eg_tau = args.tau_gate
        return cfg

    # Shared CRoPS / EntropyGate kwargs
    shared_kwargs = dict(
        key_position={
            "image_start": image_tokens[0],
            "image_end": image_tokens[-1],
        },
        input_ids_lang_prior=input_ids_lang_prior,
        aggregate_layer_fast_v=args.aggregate_layer_fast_v,
        minumum_fast_v_tokens=args.minumum_fast_v_tokens,
        aggregate_layer_text_mask=args.aggregate_layer_text_mask,
        minimum_text_tokens=args.minimum_text_tokens,
    )

    if args.method == "crops":
        return GenerationConfigContrastive(
            **common_kwargs,
            **shared_kwargs,
            lambda_lang_prior=args.lambda_lang_prior,
            alpha_stat_bias=args.alpha_stat_bias,
            beta_cutoff=args.beta_cutoff,
            max_threshold_plausibility_constraint=args.max_threshold_plausibility_constraint,
        )

    if args.method == "latent":
        return GenerationConfigLatent(
            **common_kwargs,
            **shared_kwargs,
            # EntropyGate params (reused by LEG)
            alpha_base_vis=args.alpha_base_vis,
            alpha_min_vis=args.alpha_min_vis,
            eta_vis=args.eta_vis,
            tau_gate=args.tau_gate,
            gamma_decay=args.gamma_decay,
            beta_cutoff_fixed=args.beta_cutoff_fixed,
            theta_safe=args.theta_safe,
            # HSC params
            hsc_alpha_base=args.hsc_alpha_base,
            hsc_alpha_min=args.hsc_alpha_min,
            hsc_eta=args.hsc_eta,
            hsc_tau=args.hsc_tau,
            # LEG params
            leg_hidden_layer=args.leg_hidden_layer,
            # LLH params
            llh_hidden_alpha_base=args.llh_hidden_alpha_base,
            llh_hidden_alpha_min=args.llh_hidden_alpha_min,
            llh_hidden_eta=args.llh_hidden_eta,
            llh_hidden_tau=args.llh_hidden_tau,
            # Method selector
            latent_method=args.latent_method,
        )

    # entropygate (default)
    return GenerationConfigEntropyGate(
        **common_kwargs,
        **shared_kwargs,
        alpha_base_vis=args.alpha_base_vis,
        alpha_base_txt=args.alpha_base_txt,
        alpha_min_vis=args.alpha_min_vis,
        alpha_min_txt=args.alpha_min_txt,
        eta_vis=args.eta_vis,
        eta_txt=args.eta_txt,
        tau_gate=args.tau_gate,
        gamma_decay=args.gamma_decay,
        beta_base=args.beta_base,
        beta_range=args.beta_range,
        theta_safe=args.theta_safe,
        eg_scheme=args.eg_scheme,
        beta_cutoff_fixed=args.beta_cutoff_fixed,
        theta_safe_aligned=args.theta_safe_aligned,
        time_decay_mode=args.time_decay_mode,
        alpha_time_txt=args.alpha_time_txt,
        adaptive_eta=args.adaptive_eta,
        eta_ema_momentum=args.eta_ema_momentum,
        eta_vis_offset=args.eta_vis_offset,
        eta_txt_offset=args.eta_txt_offset,
        soft_suppress=args.soft_suppress,
        soft_suppress_k=args.soft_suppress_k,
    )


def main():
    args = args_parser()
    set_reproducibility(args.seed)

    # Apply monkey-patches based on method
    if args.method == "entropygate":
        from methods.entropygate_method import patch_everything
        patch_everything()
    elif args.method == "crops":
        from methods.crops_method import patch_everything_crops
        patch_everything_crops()
    elif args.method in ("vcd", "vcd_eg"):
        from methods.vcd_method import patch_everything_vcd
        patch_everything_vcd()
    elif args.method == "latent":
        from methods.latent_method import patch_everything_latent
        patch_everything_latent()
    # vanilla: no patching needed

    # Setup logging
    eg_logger, log_file = setup_logging(args.experiment_name)
    eg_logger.info(f"EntropyGate experiment: {args.experiment_name}")
    eg_logger.info(f"Log file: {log_file}")
    eg_logger.info(f"Model: {args.model_name}")
    if args.method == "latent":
        eg_logger.info(
            f"Hyperparams: latent_method={args.latent_method} "
            f"hsc_alpha=[{args.hsc_alpha_min},{args.hsc_alpha_base}] hsc_eta={args.hsc_eta} hsc_tau={args.hsc_tau} "
            f"leg_hidden_layer={args.leg_hidden_layer} "
            f"llh_alpha=[{args.llh_hidden_alpha_min},{args.llh_hidden_alpha_base}] "
            f"gamma={args.gamma_decay} beta_cutoff={args.beta_cutoff_fixed} theta_safe={args.theta_safe}"
        )
    else:
        eg_logger.info(
            f"Hyperparams: scheme={args.eg_scheme} alpha_vis={args.alpha_base_vis} alpha_txt={args.alpha_base_txt} "
            f"alpha_min_vis={args.alpha_min_vis} alpha_min_txt={args.alpha_min_txt} "
            f"eta_vis={args.eta_vis} eta_txt={args.eta_txt} tau={args.tau_gate} "
            f"gamma={args.gamma_decay} beta_base={args.beta_base} beta_range={args.beta_range} "
            f"theta_safe={args.theta_safe}"
        )
    eg_logger.info(f"Full args: {vars(args)}")

    if args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": distributed_state.device},
        quantization_config=bnb_config,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    if args.run_chair_benchmark:
        run_chair_benchmark(model, processor, args)
    if args.run_amber_benchmark:
        run_amber_benchmark(model, processor, args)
    if args.run_mathvista_benchmark:
        run_mathvista_benchmark(model, processor, args)
    if args.run_mme_benchmark:
        run_mme_benchmark(model, processor, args)
    if args.run_mmmu_benchmark:
        run_mmmu_benchmark(model, processor, args)

def _make_system_content():
    return [{"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}]


def _build_lang_prior_inputs(processor, text, device):
    conversation = [
        {"role": "system", "content": _make_system_content()},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    out = processor.apply_chat_template(
        conversation, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device, torch.bfloat16)
    return out["input_ids"]


def _build_full_inputs(processor, image, text, device):
    conversation = [
        {"role": "system", "content": _make_system_content()},
        {"role": "user", "content": [
            {"type": "image", "url": image},
            {"type": "text", "text": text},
        ]},
    ]
    return processor.apply_chat_template(
        conversation, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device, torch.bfloat16)


def run_chair_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "EntropyGate", args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)

    chair_benchmark = ChairBenchmarkDataset(
        coco_path=args.coco_path,
        coco_file=args.coco_file,
        base_image_path=args.coco_base_image_path,
        chair_test_size=args.chair_test_size,
    )

    with distributed_state.local_main_process_first():
        test_dataset = chair_benchmark.get_test_dataset()

    prompt_text = "Please describe this image in detail"
    input_ids_lang_prior = _build_lang_prior_inputs(processor, prompt_text, distributed_state.device)
    image_token_ids = get_image_token_id(args.model_name)

    with distributed_state.split_between_processes(test_dataset) as process_test_dataset:
        generations = []
        for test_image in tqdm(process_test_dataset, desc=f"Running Chair Benchmark. Process: {distributed_state.process_index}"):
            inputs = _build_full_inputs(processor, test_image["image_path"], prompt_text, distributed_state.device)
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]

            generation_config = make_generation_config(args, image_tokens, input_ids_lang_prior)

            # VCD: inject noised pixel_values via generation_config
            # (cannot pass as model_kwargs — transformers validates them before _sample)
            if args.method in ("vcd", "vcd_eg"):
                from methods.utils.vcd_noise import add_diffusion_noise
                generation_config.pixel_values_cd = add_diffusion_noise(
                    inputs["pixel_values"], args.vcd_noise_step
                )

            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = processor.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

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


def run_amber_benchmark(model, processor, args):
    experiment_name = os.path.join(
        "experiments",
        "--".join(args.model_name.split("/")),
        "EntropyGate",
        "AMBER",
        args.experiment_name,
    )
    os.makedirs(experiment_name, exist_ok=True)

    amber_benchmark = AmberBenchmarkDataset(
        query_file=args.amber_query_file,
        image_dir=args.amber_image_dir,
        official_repo_path=args.amber_official_repo_path,
    )

    with distributed_state.local_main_process_first():
        test_dataset = amber_benchmark.get_test_dataset()

    image_token_ids = get_image_token_id(args.model_name)
    lang_prior_cache = {}

    with distributed_state.split_between_processes(test_dataset) as process_test_dataset:
        generations = []
        for sample in tqdm(
            process_test_dataset,
            desc=f"Running AMBER Benchmark. Process: {distributed_state.process_index}",
        ):
            prompt_text = sample["prompt"]
            if prompt_text not in lang_prior_cache:
                lang_prior_cache[prompt_text] = _build_lang_prior_inputs(
                    processor, prompt_text, distributed_state.device
                )
            input_ids_lang_prior = lang_prior_cache[prompt_text]

            inputs = _build_full_inputs(
                processor, sample["image_path"], prompt_text, distributed_state.device
            )
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]
            generation_config = make_generation_config(args, image_tokens, input_ids_lang_prior)

            # VCD: inject noised pixel_values via generation_config
            if args.method in ("vcd", "vcd_eg"):
                from methods.utils.vcd_noise import add_diffusion_noise
                generation_config.pixel_values_cd = add_diffusion_noise(
                    inputs["pixel_values"], args.vcd_noise_step
                )

            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = processor.decode(
                output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
            )

            try:
                del inputs["input_ids"]
            except Exception:
                pass
            del output_ids, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            generations.append(
                {
                    "id": sample["id"],
                    "response": output_text,
                }
            )

    generations = gather_object(generations)
    if distributed_state.is_main_process:
        generations_path = os.path.join(experiment_name, "amber_generations.json")
        evaluation_output_path = os.path.join(experiment_name, "amber_evaluation.txt")

        amber_benchmark.dump_generations(generations, generations_path)
        evaluation_output = amber_benchmark.evaluate(
            generations_path,
            evaluation_type=args.amber_evaluation_type,
            dump_results=True,
            evaluation_output_path=evaluation_output_path,
        )
        print("AMBER evaluation complete.")
        print(evaluation_output)

def run_mme_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "EntropyGate", "MME", args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)

    mme_dataset = load_dataset("darkyarding/MME")["test"]
    data_list = list(mme_dataset)
    image_token_ids = get_image_token_id(args.model_name)

    with distributed_state.split_between_processes(data_list) as process_data_list:
        results = []
        for sample in tqdm(process_data_list, total=len(process_data_list), desc=f"Running MME Benchmark. Process: {distributed_state.process_index}"):
            question = sample["question"]
            gt_ans = sample["answer"].lower()

            input_ids_lang_prior = _build_lang_prior_inputs(processor, question, distributed_state.device)
            inputs = _build_full_inputs(processor, sample["image"], question, distributed_state.device)
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]

            generation_config = make_generation_config(args, image_tokens, input_ids_lang_prior)

            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = processor.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            pred_ans = parse_pred_ans(output_text)

            results.append({
                "question_id": sample["question_id"],
                "category": sample["category"],
                "pred_ans": pred_ans,
                "gt_ans": gt_ans,
            })

            del output_ids, inputs, input_ids_lang_prior
            torch.cuda.empty_cache()
            gc.collect()

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
    perception_score = sum(category2avg_score[cat] for cat in eval_type_dict["Perception"])
    cognition_score = sum(category2avg_score[cat] for cat in eval_type_dict["Cognition"])

    with open(os.path.join(experiment_name, 'mme_results.txt'), "a") as f:
        f.write(f"{args}\n")
        f.write("=========== Perception ===========\n")
        f.write(f"total score: {perception_score:.2f}\n\n")
        for category in eval_type_dict["Perception"]:
            f.write(f"\t {category}  score: {category2avg_score[category]:.2f}\n")
        f.write("\n=========== Cognition ===========\n")
        f.write(f"total score: {cognition_score:.2f}\n\n")
        for category in eval_type_dict["Cognition"]:
            f.write(f"\t {category}  score: {category2avg_score[category]:.2f}\n")

    print("MME evaluation complete. Results saved.")

def run_mathvista_benchmark(model, processor, args):
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "EntropyGate", "MathVista", args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)

    data_list = load_dataset('AI4Math/MathVista', split='testmini')
    image_token_ids = get_image_token_id(args.model_name)
    generations = []

    with distributed_state.split_between_processes(data_list) as process_data_list:
        for problem in tqdm(process_data_list, desc=f"Running MathVista Benchmark. Process: {distributed_state.process_index}"):
            problem_decoded_image = problem['decoded_image']
            problem.pop('decoded_image')
            query = problem['query']

            input_ids_lang_prior = _build_lang_prior_inputs(processor, query, distributed_state.device)
            inputs = _build_full_inputs(processor, problem_decoded_image, query, distributed_state.device)
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]

            generation_config = make_generation_config(args, image_tokens, input_ids_lang_prior)

            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = processor.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

            generations.append({
                "pid": problem['pid'],
                "query": query,
                "response": output_text,
            })

    generations = gather_object(generations)
    if distributed_state.is_main_process:
        output_file_path = os.path.join(experiment_name, "mathvista_generations.jsonl")
        with open(output_file_path, 'w') as f:
            json.dump(generations, f, indent=4)


def run_mmmu_benchmark(model, processor, args):
    if not _HAS_MMMU:
        raise ImportError("benchmark.mmmu_utils not found. Please ensure mmmu_utils.py is available to run MMMU benchmark.")
    experiment_name = os.path.join("experiments", "--".join(args.model_name.split("/")), "MMMU", "EntropyGate", args.experiment_name)
    os.makedirs(experiment_name, exist_ok=True)

    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset("MMMU/MMMU", subject, split='validation')
        sub_dataset_list.append(sub_dataset)
    dataset = concatenate_datasets(sub_dataset_list)
    data_list = list(dataset)
    image_token_ids = get_image_token_id(args.model_name)

    with distributed_state.split_between_processes(data_list) as process_data_list:
        out_samples = dict()
        for sample in tqdm(process_data_list, total=len(process_data_list), desc=f"Running MMMU Benchmark. Process: {distributed_state.process_index}"):
            sample = process_single_sample(sample)
            sample = construct_prompt(sample)

            prompt = sample['final_input_prompt']
            image = sample['image']

            input_ids_lang_prior = _build_lang_prior_inputs(processor, prompt, distributed_state.device)
            inputs = _build_full_inputs(processor, image, prompt, distributed_state.device)
            image_tokens = np.where(inputs["input_ids"].cpu().numpy() == image_token_ids)[1]

            generation_config = make_generation_config(args, image_tokens, input_ids_lang_prior)

            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=generation_config)
            output_text = processor.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            out_samples[sample['id']] = output_text

    output_path = os.path.join(experiment_name, 'mmmu_answers.json')
    with open(output_path, 'w') as f:
        json.dump(out_samples, f, indent=4)

    results = evaluate_mmmu(output_path, args.mmmu_answer_file_path)
    with open(os.path.join(experiment_name, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
