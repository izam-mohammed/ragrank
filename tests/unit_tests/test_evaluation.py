from ragrank import evaluate


def test_evaluation():
    result = evaluate()
    assert (
        result == "evaluation successful"
    ), "Result should match the expected output"
