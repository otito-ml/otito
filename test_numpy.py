from otito.metrics import load_metric
import numpy as np
import timeit

accuracy_score = load_metric(metric="Accuracy", package="numpy", parse_input=True)

start_time = timeit.default_timer()

result = accuracy_score(
    y_observed=np.array([1, 1, 0]),
    y_predicted=np.array([1, 0, 0]),
    sample_weights=np.array([0.6, 0.3, 0.1]),
)

print(f"Otito (Numpy): {timeit.default_timer()-start_time}: Result: {result}")
