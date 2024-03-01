class RagRankError(Exception): 
    """Exception class for ragrank"""


class EvaluationError(RagRankError):
    """Evaluation Exception for ragrank"""
    def __init__(self, message="Error during evaluation."):
        self.message = message
        super().__init__(self.message)
