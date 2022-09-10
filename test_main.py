from otito.metrics.numpy import Accuracy
import numpy as np
import timeit

start_time = timeit.default_timer()
accuracy_score = Accuracy(parse_input=False)
result = accuracy_score(
    y_observed=np.array([1, 1, 0]),
    y_predicted=np.array([1, 0, 0]),
    sample_weights=np.array([0.6, 0.3, 0.1]),
)
print(f"Otito: {timeit.default_timer()-start_time}: Result: {result}")
