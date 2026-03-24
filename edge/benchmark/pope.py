"""POPE (Polling-based Object Probing Evaluation) benchmark.

Evaluates object hallucination via yes/no questions about object presence.
Three splits: random, popular, adversarial.
"""

import json
import os
import re
from collections import defaultdict


class PopeBenchmarkDataset:
    def __init__(self, pope_path: str, coco_image_dir: str, pope_splits: list[str] | None = None):
        self.pope_path = pope_path
        self.coco_image_dir = coco_image_dir
        self.pope_splits = pope_splits or ["random", "popular", "adversarial"]

    def _load_split(self, split: str) -> list[dict]:
        candidates = [
            f"coco_pope_{split}.json", f"coco_pope_{split}.jsonl",
            f"pope_{split}.json", f"pope_{split}.jsonl",
        ]
        file_path = None
        for candidate in candidates:
            path = os.path.join(self.pope_path, candidate)
            if os.path.exists(path):
                file_path = path
                break

        if file_path is None:
            raise FileNotFoundError(
                f"Cannot find POPE {split} file. "
                f"Tried: {[os.path.join(self.pope_path, c) for c in candidates]}"
            )

        samples = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                image_name = record.get("image", "")
                image_path = os.path.join(self.coco_image_dir, image_name)
                samples.append({
                    "question_id": record.get("question_id", len(samples)),
                    "image_path": image_path,
                    "image_name": image_name,
                    "prompt": record["text"],
                    "label": record["label"].strip().lower(),
                    "split": split,
                })
        return samples

    def get_test_dataset(self) -> list[dict]:
        all_samples = []
        for split in self.pope_splits:
            split_samples = self._load_split(split)
            all_samples.extend(split_samples)
            print(f"POPE {split}: loaded {len(split_samples)} samples")
        print(f"POPE total: {len(all_samples)} samples")
        return all_samples

    @staticmethod
    def dump_generations(results: list[dict], results_path: str):
        with open(results_path, "w") as f:
            for item in results:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    @staticmethod
    def _extract_yesno(response: str) -> str:
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        response_lower = response.strip().lower()
        if not response_lower:
            return "unknown"

        first_word = response_lower.split()[0].rstrip(".,!;:")
        if first_word in ("yes", "no"):
            return first_word

        has_yes = "yes" in response_lower
        has_no = "no" in response_lower
        if has_yes and not has_no:
            return "yes"
        if has_no and not has_yes:
            return "no"

        return "unknown"

    def evaluate(self, results_path: str, dump_results: bool = True) -> dict:
        with open(results_path, "r") as f:
            results = [json.loads(line) for line in f if line.strip()]

        split_metrics = {}
        all_correct = 0
        all_total = 0
        split_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

        for item in results:
            split = item.get("split", "unknown")
            label = item["label"].strip().lower()
            pred = self._extract_yesno(item["response"])

            is_correct = pred == label
            if is_correct:
                all_correct += 1
            all_total += 1

            counts = split_counts[split]
            if pred == "yes" and label == "yes":
                counts["tp"] += 1
            elif pred == "yes" and label == "no":
                counts["fp"] += 1
            elif pred == "no" and label == "yes":
                counts["fn"] += 1
            else:
                counts["tn"] += 1

        for split, counts in split_counts.items():
            tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
            total = tp + fp + fn + tn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            split_metrics[split] = {
                "accuracy": accuracy, "precision": precision,
                "recall": recall, "f1": f1, "total": total,
                "yes_ratio": (tp + fp) / total if total > 0 else 0.0,
            }

        overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
        report = {"overall_accuracy": overall_accuracy, "overall_total": all_total, "splits": split_metrics}

        print("\n========== POPE Evaluation ==========")
        print(f"Overall Accuracy: {overall_accuracy * 100:.1f}% ({all_correct}/{all_total})")
        for split in sorted(split_metrics.keys()):
            m = split_metrics[split]
            print(f"\n--- {split.upper()} ---")
            print(f"  Accuracy:  {m['accuracy'] * 100:.1f}%")
            print(f"  Precision: {m['precision'] * 100:.1f}%")
            print(f"  Recall:    {m['recall'] * 100:.1f}%")
            print(f"  F1:        {m['f1'] * 100:.1f}%")
            print(f"  Yes ratio: {m['yes_ratio'] * 100:.1f}%")
        print("=====================================\n")

        if dump_results:
            metrics_path = os.path.join(os.path.dirname(results_path), "pope_results.json")
            with open(metrics_path, "w") as f:
                json.dump(report, f, indent=2)

        return report
