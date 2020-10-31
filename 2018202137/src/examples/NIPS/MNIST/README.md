# MNIST Addition examples

These are toy examples showing the integration of high-level reasoning and low-level perception. The tasks are based on the digit recognition task of the MNIST dataset.

## Single-digit MNIST Addition
In this experiment, the task is to classify the sum of two MNIST digits. The ``generate_data.py`` was used to generate the train and test sets by randomly pairing digits in the respective sets of the original dataset.

```bash
python single_digit/run.py
```

## Multi-digit MNIST Addition
In this experiment, the task is to classify the sum of two lists MNIST digits representing multi-digit numbers. The ``generate_data.py`` was used to generate the train and test sets by randomly grouping the digits in the respective sets of the original dataset.

```bash
python single_digit/run.py
```
