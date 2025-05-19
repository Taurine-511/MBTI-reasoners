from reasoners import WorldModel, LanguageModel
from typing import NamedTuple
import copy


class ReasoningState(NamedTuple):
    step_idx: int
    action_history: list[str]
    end: bool


ReasoningAction = str


class ReasoningWorldModel(WorldModel):
    def __init__(
        self,
        base_model: LanguageModel,
        prompt: dict,
        max_steps: int = 4,
        batch_size: int = 1,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size

    def init_state(self) -> ReasoningState:
        return ReasoningState(step_idx=0, action_history=[], end=False)

    def step(
        self, state: ReasoningState, action: ReasoningAction
    ) -> tuple[ReasoningState, dict]:
        state = copy.deepcopy(state)
        if self.base_model.tokenizer.eos_token not in action:
            state = ReasoningState(
                state.step_idx + 1, state.action_history + [action], False
            )
        else:
            state = ReasoningState(state.step_idx + 1, state.action_history, True)
        return state, {}

    def is_terminal(self, state: ReasoningState) -> bool:
        return state.end or state.step_idx >= self.max_steps
