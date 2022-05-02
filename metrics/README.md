# metrics
Module used to calculate nested NER metrics.

1. Install the nestednereval library: 

```bash
pip install git+https://github.com/matirojasg/nestednereval.git
```

2. Put the prediction files in the respective folders.

3. Execute `python main.py`.

4. Output:

```
Nested metrics of model: flair in dataset: wl.

Standard metric Precision: 82.06        Recall: 83.94   F1-Score: 82.99 support: 8837
Flat metric     Precision: 85.11        Recall: 84.23   F1-Score: 84.67 support: 4655
Inner metric    Precision: 85.1 Recall: 89.91   F1-Score: 87.44 support: 2350
Outer metric    Precision: 70.97        Recall: 75.55   F1-Score: 73.19 support: 1832
Nested metric   Precision: 78.89        Recall: 83.62   F1-Score: 81.18 support: 4182
Nesting metric  Precision: 58.86        Recall: 64.36   F1-Score: 61.49 support: 1832


Nested metrics of model: flair in dataset: clinical_trials.

Standard metric Precision: 86.01        Recall: 86.58   F1-Score: 86.29 support: 8940
Flat metric     Precision: 86.37        Recall: 86.41   F1-Score: 86.39 support: 6607
Inner metric    Precision: 88.06        Recall: 91.63   F1-Score: 89.81 support: 1231
Outer metric    Precision: 81.5 Recall: 81.94   F1-Score: 81.72 support: 1102
Nested metric   Precision: 85.01        Recall: 87.06   F1-Score: 86.02 support: 2333
Nesting metric  Precision: 71.21        Recall: 74.5    F1-Score: 72.82 support: 1102


Nested metrics of model: flert in dataset: wl.

Standard metric Precision: 79.42        Recall: 82.66   F1-Score: 81.01 support: 8837
Flat metric     Precision: 81.67        Recall: 83.18   F1-Score: 82.42 support: 4655
Inner metric    Precision: 81.3 Recall: 88.81   F1-Score: 84.89 support: 2350
Outer metric    Precision: 71.22        Recall: 73.47   F1-Score: 72.33 support: 1832
Nested metric   Precision: 77.02        Recall: 82.09   F1-Score: 79.48 support: 4182
Nesting metric  Precision: 57.65        Recall: 62.94   F1-Score: 60.18 support: 1832


Nested metrics of model: flert in dataset: clinical_trials.

Standard metric Precision: 88.73        Recall: 90.44   F1-Score: 89.57 support: 8940
Flat metric     Precision: 89.55        Recall: 90.54   F1-Score: 90.04 support: 6607
Inner metric    Precision: 89.58        Recall: 92.93   F1-Score: 91.23 support: 1231
Outer metric    Precision: 83.03        Recall: 87.02   F1-Score: 84.98 support: 1102
Nested metric   Precision: 86.47        Recall: 90.14   F1-Score: 88.27 support: 2333
Nesting metric  Precision: 74.34        Recall: 79.67   F1-Score: 76.92 support: 1102


Nested metrics of model: roberta in dataset: wl.

Standard metric Precision: 80.43        Recall: 78.76   F1-Score: 79.58 support: 8837
Flat metric     Precision: 82.43        Recall: 79.31   F1-Score: 80.84 support: 4655
Inner metric    Precision: 83.81        Recall: 86.13   F1-Score: 84.95 support: 2350
Outer metric    Precision: 70.68        Recall: 67.9    F1-Score: 69.27 support: 1832
Nested metric   Precision: 78.28        Recall: 78.14   F1-Score: 78.21 support: 4182
Nesting metric  Precision: 56.21        Recall: 56.33   F1-Score: 56.27 support: 1832


Nested metrics of model: roberta in dataset: clinical_trials.

Standard metric Precision: 88.07        Recall: 86.21   F1-Score: 87.13 support: 8940
Flat metric     Precision: 88.23        Recall: 86.59   F1-Score: 87.4  support: 6607
Inner metric    Precision: 90.5 Recall: 87.41   F1-Score: 88.93 support: 1231
Outer metric    Precision: 84.42        Recall: 82.58   F1-Score: 83.49 support: 1102
Nested metric   Precision: 87.6 Recall: 85.13   F1-Score: 86.35 support: 2333
Nesting metric  Precision: 74.98        Recall: 72.05   F1-Score: 73.48 support: 1102


Nested metrics of model: spanish-bert in dataset: wl.

Standard metric Precision: 77.44        Recall: 74.5    F1-Score: 75.94 support: 8837
Flat metric     Precision: 79.4 Recall: 75.25   F1-Score: 77.27 support: 4655
Inner metric    Precision: 82.33        Recall: 81.66   F1-Score: 81.99 support: 2350
Outer metric    Precision: 66.02        Recall: 63.43   F1-Score: 64.7  support: 1832
Nested metric   Precision: 75.31        Recall: 73.67   F1-Score: 74.48 support: 4182
Nesting metric  Precision: 52.49        Recall: 51.2    F1-Score: 51.84 support: 1832


Nested metrics of model: spanish-bert in dataset: clinical_trials.

Standard metric Precision: 86.34        Recall: 82.57   F1-Score: 84.41 support: 8940
Flat metric     Precision: 86.58        Recall: 83.29   F1-Score: 84.9  support: 6607
Inner metric    Precision: 87.41        Recall: 82.94   F1-Score: 85.12 support: 1231
Outer metric    Precision: 83.63        Recall: 77.86   F1-Score: 80.64 support: 1102
Nested metric   Precision: 85.64        Recall: 80.54   F1-Score: 83.01 support: 2333
Nesting metric  Precision: 71.56        Recall: 65.97   F1-Score: 68.65 support: 1102
```