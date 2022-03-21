from omegaconf import OmegaConf

def hydra_to_sacred(input):
    input_transformed = dict()
    input_transformed["named_configs"] = OmegaConf.to_object(input["named_configs"])
    input_transformed["config_updates"] = OmegaConf.to_object(input["config_updates"])

    if "command_name" in input.keys():
        input_transformed["command_name"] = str(input["command_name"])

    return input_transformed