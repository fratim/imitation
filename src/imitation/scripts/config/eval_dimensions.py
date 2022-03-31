import sacred

from imitation.scripts.common import common, demonstrations


eval_dimensions_ex = sacred.Experiment(
    "eval_dimensions",
    ingredients=[common.common_ingredient, demonstrations.demonstrations_ingredient],
)

@eval_dimensions_ex.config
def eval_dimensions_defaults():
    test = 10

@eval_dimensions_ex.named_config
def seals_mountain_car():
    test = 20
