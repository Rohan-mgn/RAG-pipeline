# stage7_evaluate/__init__.py
from .errors import ConfigError, GoldenSetError, Stage7Error
from .golden import GoldenQuestion, golden_fingerprint, load_golden
from .judge import judge_question
from .metrics import (aspect_relevance, context_precision, context_recall,
                      fact_coverage, faithfulness, mark_relevant)
from .report import aggregate, build_report, diff_reports, save_report
from .runner import EvalConfig, EvalRunner, QuestionResult