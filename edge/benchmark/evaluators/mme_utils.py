"""MME benchmark utilities — re-exported from the original codebase."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from benchmark.evaluators.mme.utils import eval_type_dict, parse_pred_ans
