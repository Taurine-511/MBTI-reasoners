from my_reasoning.world_model import ReasoningWorldModel
from my_reasoning.search_config.baseline import ReasoningConfig

import json
import torch
from reasoners import Reasoner
from reasoners.lm import HFModel
from reasoners.algorithm import BeamSearch


def main():
    model_name = "llm-jp/llm-jp-3-13b-instruct3"
    model = HFModel(
        model_name,
        model_name,
        device=torch.device("cuda:0"),
        max_batch_size=32,
        max_new_tokens=512,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    with open("data/answers_ENFP_nemotron.json", "r") as f:
        dataset = json.load(f)
    with open("prompts/base_prompt.json", "r") as f:
        prompt = json.load(f)

    example = dataset[0]
    world_model = ReasoningWorldModel(base_model=model, prompt=prompt)
    config = ReasoningConfig(base_model=model, prompt=prompt)
    algorithm = BeamSearch(beam_size=4, max_depth=7)
    reasoner = Reasoner(
        world_model=world_model, search_config=config, search_algo=algorithm
    )
    result = reasoner(example)
    print(result)


if __name__ == "__main__":
    main()
