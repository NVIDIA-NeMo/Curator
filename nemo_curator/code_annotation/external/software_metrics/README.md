# software_metrics


## Usage

```python
LANGUAGE = 'Python'
CODE = "print(2)\nx=3\n#comment\n\n\n\n\n"

metrics = software_metrics.get_all_metrics(LANGUAGE, CODE)

# This will return a python dict[str, float] that looks like this:
# {
#  'cyclomatic': 1.0, 'cognitive_complexity': nan, 'exits average': nan,
#  'maintainability_index': 73.97468378825788, 'halstead_difficulty': 0.5,
#  '# comments': 1.0, '# comments per space': 1.0, '# blank lines': 4.0,
#  '# blank lines per space': 4.0, '# args average': 0.0, 'functions/closures per space': 0.0
# }
```
