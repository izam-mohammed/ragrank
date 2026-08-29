(jury-and-pairwise)=
# When One Judge Is Not Enough

A language model asked the same question twice does not always answer
the same way, and its absolute scores are poorly calibrated. Ragrank
offers three ways to deal with that.

## Ask several judges

```python
from ragrank.metric import Jury

panel = Jury(judges=[gpt_judge, claude_judge, local_judge])
```

A committee is more stable than any one member. Median by default, mean
if you prefer, and the spread is reported:

```python
result.results[0][0].metadata["disagreement"]
```

That number is worth reading on its own. Where the panel argues loudly
is usually where your rubric is ambiguous, which is a problem no amount
of averaging will fix.

A judge that fails is dropped rather than sinking the panel. Every
judge's verdict is recorded individually, including judges that share a
name.

## Compare instead of scoring

```python
from ragrank.metric import Pairwise

better_than_baseline = Pairwise(baseline_field="reference")
```

Models are markedly better at "which of these two is better" than at
"rate this from 0 to 1", and comparing two systems is usually the
question you actually have.

**Position bias is real.** Judges favour whichever answer they see
first. Ragrank runs every comparison twice with the order swapped, and
a verdict that flips is reported as a tie rather than as a win for
whichever ordering happened to be asked first:

```python
result.results[0][0].metadata
# {"forward": 1.0, "reverse": 0.0, "position_bias": True}
```

A judge that always picks the first answer scores 0.5, not 1.0.

## Ask the same judge repeatedly

```python
from ragrank.evaluation import RunConfig

evaluate(data, run_config=RunConfig(repetitions=5, reducer="median"))
```

The simplest option, and it makes the variance visible rather than
hiding it behind a single sample:

```python
result.results[0][0].metadata["repetitions"]        # [0.9, 0.5, 0.7, 0.6, 0.8]
result.results[0][0].metadata["repetition_spread"]  # 0.158
```

Off by default, because it multiplies the cost of a run by the number
of repetitions.
