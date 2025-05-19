from reasoners import SearchConfig, LanguageModel
from ..world_model import ReasoningState, ReasoningAction


# ToTをベースとする報酬設計
# log-prob + self-eval
class ReasoningConfig(SearchConfig):
    def __init__(
        self,
        base_model: LanguageModel,
        prompt: dict,
        temperature: float = 0.8,
        n_candidate: int = 4,
    ):
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.temperature = temperature
        self.n_candidate = n_candidate
        self.example = None

    def get_actions(self, state: ReasoningState) -> list[ReasoningAction]:
        prompts = (
            self.prompt["input"]
            .replace("<question>", self.example["generated_question"])
            .replace("<action>", "".join(state.action_history))
        )
        # 現状はトークンレベルでの探索を行う
        outputs = self.base_model.generate(
            [prompts],
            num_return_sequences=self.n_candidate,
            max_new_tokens=32,
            temperature=self.temperature,
            do_sample=True,
            hide_input=True,
        ).text
        outputs = [output.split("\n")[0] for output in outputs]
        return list(dict.fromkeys(outputs))

    def fast_reward(
        self, state: ReasoningState, action: ReasoningAction
    ) -> tuple[float, dict]:
        prompts = (
            self.prompt["input"]
            .replace("<question>", self.example["generated_question"])
            .replace("<action>", "".join(state.action_history))
        )
        intuition = self.base_model.get_loglikelihood(prompts, [prompts + action])[0]
        self_eval_prompt = (
            self.prompt["self-eval"]
            .replace("<question>", self.example["generated_question"])
            .replace("<action>", "".join(state.action_history))
        )
        self_eval = self.base_model.get_loglikelihood(
            self_eval_prompt, [self_eval_prompt + "yes"]
        )[0]
        return intuition + self_eval, {"intuition": intuition, "self_eval": self_eval}

    def reward(self, state, action, **kwargs) -> tuple[float, dict]:
        return kwargs["intuition"] + kwargs["self_eval"], kwargs
