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
    y_observed=[1, 1, 0, 1],
    y_predicted=[1, 0, 0, 1]
)
print(result)
>>> 0.75
```

# Contributing
\# TODO write this section
