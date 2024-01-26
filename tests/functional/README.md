### Basic concept of the functional tests

Here is added the flexible method for make function testing<br>
of the OCCA transpiler tool. It's based on the Json configuration<br>
of the test suite where there is possibility to provide set of tests cases<br>
Each test case can be address for one of the pipeline step:
* Normalization
* Transpilation
* Normalization and Transpilation

Each of pipeline step has the Json specific structure to be described: <br>

```json
    {
        "action": "transpilier",
        "action_config": {
            "backend": "cuda",
            "source": "",
            "includes" : [],
            "defs": [],
            "launcher": ""
        },
        "reference": ""
     },
     {
        "action": "normalizer",
        "action_config": {
            "source": ""
        },
        "reference": ""
     },
     {
        "action": "transpile_and_normalize",
        "action_config": {
            "backend": "cuda",
            "source": "",
            "includes" : [],
            "defs": [],
            "launcher": ""
        },
        "reference": ""
     }
```

For the transpiling backend field contains enum value:

* `cuda`
* `openmp`
