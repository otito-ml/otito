from otito.metrics import load_metric
import torch as pt
import timeit

accuracy_score = load_metric(metric="Accuracy", package="pytorch", validate_input=True)

start_time = timeit.default_timer()
result = accuracy_score(
    y_observed=pt.tensor([1, 1, 0]),
    y_predicted=pt.tensor([1, 0, 0]),
    sample_weights=pt.tensor([0.6, 0.3, 0.1]),
)
print(f"Otito (Pytorch): {timeit.default_timer()-start_time}: Result: {result}")
