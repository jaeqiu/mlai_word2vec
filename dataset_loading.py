import json

def lazy_load_dataset(jsonpath = "Musical_Instruments_5.json", colselect="reviewText", n_lines = None):
    with open(jsonpath) as f:
        for i, line in enumerate(f):
            if n_lines is None or i <= n_lines: 
                obj = json.loads(line)
                yield(obj[colselect])
            else:
                break