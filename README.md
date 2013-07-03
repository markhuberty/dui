dui
===


## Dependencies:
- pandas
- dedupe

## Run process:

``` python
# To clean the data:
# Assumes a file in ./data/ called alcdata.csv
python clean.py

# Then dedupe:
# The path to the input should be ../data/dedupe_alc_input.csv 
# if you ran clean on it first
python dedupe_dui.py <path_to_input_file>

```

