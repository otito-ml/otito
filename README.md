# Otito

**Otito** is a tool that provides a suite of model agnostic methods to evaluate the behaviour and performance of ml models.


## Getting Started

### Installing `Otito`

```
> pip install otito
```

### Quickstart

#### Example

```
from otito.metrics import accuracy_score

result = accuracy_score(
    y=[1, 1, 0, 1],
    y_predicted=[1, 0, 0, 1]
)
print(result)
>>> 0.75
```

# Contributing

Every Metric needs the following
- A function in `metrics/` directory that can be called directly by the user
- Metric definition(s) located in the same directory
- A set of input criteria that is specified in the documentation
- A validator class that will enforce the specified criteria
- An entry in the `Metrics` section of the documentation providing:
    - Metric function definition and default values
    - A description of the arguments
    - A description of the returned value
    - Latex code outlining the metric definition (if practical)
    - Links to any papers introducing/investigating the metric.
    - Examples of usage
- [Optional] A blog post hosted on the documentation paper demonstrating an example of a real world scenario that the metrics will be if use.
