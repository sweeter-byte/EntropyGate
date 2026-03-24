"""AMBER benchmark for multi-dimensional hallucination evaluation."""

import json
import os
import subprocess
import sys
from pathlib import Path


class AmberBenchmarkDataset:
    def __init__(self, query_file: str, image_dir: str, official_repo_path: str,
                 default_prompt: str = "Please describe this image in detail"):
        self.query_file = query_file
        self.image_dir = image_dir
        self.official_repo_path = official_repo_path
        self.default_prompt = default_prompt
        self.samples = self._load_query_data()

    def _load_query_data(self) -> list[dict]:
        with open(self.query_file, "r") as f:
            data = json.load(f)

        samples = []
        for raw_sample in data:
            sample_id = self._get_first(raw_sample, ["id", "sample_id", "question_id"])
            if sample_id is None:
                raise KeyError(f"Could not find identifier in AMBER record keys: {list(raw_sample.keys())}")

            prompt = self._get_first(raw_sample, ["query", "prompt", "question", "instruction"], self.default_prompt)
            image_path = self._resolve_image_path(raw_sample, sample_id)

            samples.append({"id": int(sample_id), "prompt": prompt, "image_path": image_path, "raw_sample": raw_sample})
        return samples

    def _resolve_image_path(self, raw_sample: dict, sample_id: int) -> str:
        image_value = self._get_first(raw_sample, ["image", "image_path", "img", "filename", "file_name"])

        candidate_names = []
        if image_value is not None:
            image_value = str(image_value)
            if os.path.isabs(image_value) and os.path.exists(image_value):
                return image_value
            candidate_names.append(image_value)

        candidate_names.extend([f"AMBER_{sample_id}.jpg", f"AMBER_{sample_id}.jpeg", f"AMBER_{sample_id}.png"])
        for candidate_name in candidate_names:
            candidate_path = os.path.join(self.image_dir, candidate_name)
            if os.path.exists(candidate_path):
                return candidate_path

        raise FileNotFoundError(f"Could not resolve image for AMBER sample {sample_id}.")

    @staticmethod
    def _get_first(sample: dict, candidate_keys: list[str], default=None):
        for key in candidate_keys:
            if key in sample:
                return sample[key]
        return default

    def get_test_dataset(self) -> list[dict]:
        return self.samples

    @staticmethod
    def dump_generations(results: list[dict], results_path: str):
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    def evaluate(self, results_path: str, evaluation_type: str = "g",
                 dump_results: bool = True, evaluation_output_path: str | None = None) -> str:
        inference_script = Path(self.official_repo_path) / "inference.py"
        if not inference_script.exists():
            raise FileNotFoundError(f"Could not find AMBER evaluator script at {inference_script}")

        results_path = str(Path(results_path).resolve())
        cmd = [sys.executable, str(inference_script), "--inference_data", results_path, "--evaluation_type", evaluation_type]

        proc = subprocess.run(cmd, cwd=self.official_repo_path, capture_output=True, text=True, check=False)
        output = proc.stdout
        if proc.stderr:
            output = f"{output}\n{proc.stderr}".strip()

        if dump_results and evaluation_output_path:
            with open(evaluation_output_path, "w") as f:
                f.write(output)
                if output and not output.endswith("\n"):
                    f.write("\n")

        if proc.returncode != 0:
            raise RuntimeError(f"AMBER evaluation failed with exit code {proc.returncode}.\n{output}")

        return output
