from dagster import Definitions
from .assets import test_metrics, f1_barchart, pr_curve_fe, roc_curve_fe

defs = Definitions(assets=[test_metrics, f1_barchart, pr_curve_fe, roc_curve_fe])
