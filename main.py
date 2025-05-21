import argparse
import json
import torch
import pickle
import os
from tqdm import tqdm
from reasoners import Reasoner
from reasoners.lm import HFModel, OpenAIModel
from reasoners.algorithm import BeamSearch
from my_reasoning.world_model import ReasoningWorldModel
from my_reasoning.search_config.baseline import ReasoningConfig


def load_model(model_name, device="auto"):
    if "gpt" in model_name:
        model = OpenAIModel(
            model=model_name,
        )
    else:
        model = HFModel(
            model_name,
            model_name,
            device=torch.device(device),
            max_batch_size=32,
            max_new_tokens=512,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    return model


def load_path(path):
    with open(path, "r") as f:
        result = json.load(f)
    return result


def main(args):
    model = load_model(args.model_name, args.device)
    dataset = load_path(args.data_path)
    prompt = load_path(args.prompt_path)
    world_model = ReasoningWorldModel(base_model=model, prompt=prompt)
    config = ReasoningConfig(base_model=model, prompt=prompt)
    algorithm = BeamSearch(beam_size=args.beam_size, max_depth=args.max_depth)
    reasoner = Reasoner(
        world_model=world_model, search_config=config, search_algo=algorithm
    )

    # 完了済みのquestionを取得
    output_dir = os.path.join(args.log_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    finished_questions = set()
    for filename in os.listdir(output_dir):
        finished_questions.add(filename)

    # 推論開始
    for example in tqdm(dataset):
        filename = f"{example['question_id']}.pkl"
        if args.resume and filename in finished_questions:
            continue

        result = reasoner(example)
        with open(os.path.join(output_dir, filename), "wb") as f:
            pickle.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Reasoner with configurable settings."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llm-jp/llm-jp-3-13b-instruct3",
        help="Name of the HF model to use.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/answers_ENFP_nemotron.json",
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts/base_prompt.json",
        help="Path to the prompt JSON file.",
    )
    parser.add_argument(
        "--beam_size", type=int, default=4, help="Beam size for beam search."
    )
    parser.add_argument(
        "--max_depth", type=int, default=7, help="Maximum depth for beam search."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the model on."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save output logs and results.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default_experiment",
        help="Name of the experiment for tracking purposes.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from a previous experiment."
    )
    args = parser.parse_args()
    main(args)
