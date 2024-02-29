class RagRankError(Exception): ...


class ModelInitializationError(RagRankError):
    def __init__(self, message="Error during model initialization."):
        self.message = message
        super().__init__(self.message)


class EvaluationError(RagRankError):
    def __init__(self, message="Error during evaluation."):
        self.message = message
        super().__init__(self.message)
