"""MME benchmark utilities.

Inlined from the project-root ``benchmark/evaluators/mme/utils.py`` so that the
``edge/`` package is self-contained and can run without the root ``benchmark/``
package being present on the deployment server.
"""

import re

from torch.utils.data import Dataset


class MMEDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        length = 0
        for _ in self.ds:
            length += 1
        return length

    def __getitem__(self, index):
        row = self.ds[index]
        image = (row['image']).convert("RGB")
        question_id = row['question_id']
        question = row['question']
        category = row['category']
        return {
            "image": image,
            "answer": row["answer"],
            "question_id": question_id,
            "question": question,
            "category": category,
        }


# Predefined category classification
eval_type_dict = {
    "Perception": [
        "existence", "count", "position", "color", "posters",
        "celebrity", "scene", "landmark", "artwork", "OCR"
    ],
    "Cognition": [
        "commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"
    ]
}


def parse_pred_ans(pred_ans):
    """Parse model output into 'yes' or 'no'."""
    # Strip <think>...</think> blocks from reasoning models (e.g. R1-Onevision)
    pred_ans = re.sub(r'<think>.*?</think>', '', pred_ans, flags=re.DOTALL).strip()
    pred_ans = pred_ans.lower().strip().replace(".", "")
    if pred_ans in ["yes", "no"]:
        return pred_ans
    elif pred_ans.startswith("y"):
        return "yes"
    elif pred_ans.startswith("n"):
        return "no"
    return "other"
