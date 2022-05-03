# metrics
Module used to calculate nested NER metrics.

1. Install the nestednereval library: 

```bash
pip install git+https://github.com/matirojasg/nestednereval.git
```

2. The library metrics requires the real and predicted entities in a list as follows:

```
entities = [{"real": [("Body Part", 2, 2), ("Disease", 0, 2)], "pred": [("Body Part", 2, 2), ("Disease", 0, 2)]},
{"real": [("Body Part", 2, 2), ("Disease", 0, 2)], "pred": [("Disease", 0, 2)]},
{"real": [("Medication", 1,1)], "pred": [("Medication", 1,1)]}
]

```

If the predictions come from the MLC model, such as the example of the mlc-flair folder, the library provides a function to create the required format. An example is shown in the main.py file.


3. Execute `python main.py`.

4. The output is the following:

```
Nested metrics of model: flair in dataset: clinical_trials.

Standard metric Precision: 86.01        Recall: 86.58   F1-Score: 86.29support: 8940
Flat metric     Precision: 86.37        Recall: 86.41   F1-Score: 86.39support: 6607
Inner metric    Precision: 88.06        Recall: 91.63   F1-Score: 89.81support: 1231
Outer metric    Precision: 81.5 Recall: 81.94   F1-Score: 81.72 support: 1102
Nested metric   Precision: 85.01        Recall: 87.06   F1-Score: 86.02support: 2333
Nesting metric  Precision: 71.21        Recall: 74.5    F1-Score: 72.82support: 1102

```