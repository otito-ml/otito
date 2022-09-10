from otito.metrics.pytorch import Accuracy
import torch as pt
import timeit

start_time = timeit.default_timer()
accuracy_score = Accuracy()
result = accuracy_score(
    y_observed=pt.tensor([1, 1, 0]),
    y_predicted=pt.tensor([1, 0, 0]),
    sample_weights=pt.tensor([0.6, 0.3, 0.1]),
)
print(f"Otito (Pytorch): {timeit.default_timer()-start_time}: Result: {result}")
