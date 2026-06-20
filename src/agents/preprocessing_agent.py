from __future__ import annotations

from src.agents.base import AgentContext, BaseAgent
from src.preprocessing.enhancer import ImagePreprocessor


class PreprocessingAgent(BaseAgent):
    name = "PreprocessingAgent"

    def __init__(self, preprocessor: ImagePreprocessor) -> None:
        self.preprocessor = preprocessor

    def execute(self, ctx: AgentContext) -> AgentContext:
        image, steps = self.preprocessor.process(ctx.raw_image)
        ctx.image = image
        ctx.preprocessing_steps = steps
        return ctx

    def _summary(self, ctx: AgentContext) -> dict:
        return {"steps": ctx.preprocessing_steps}
