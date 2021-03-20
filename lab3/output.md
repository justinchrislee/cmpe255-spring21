| Experiment | Accuracy | Confusion Matrix | Comment |
|--------------|----------|------------------|---------|
| Solution 1   | xxxxxxx  | xxxxxx |  xxxxx |
| Solution 2   | 0.7864583333333334  | [[116 14]  [27 35]] | With 'glucose' seeming to play a role in increasing the accuracy, another adjustment made was adding 'bmi' as a feature along with 'glucose'. Also, adjusted the random_state parameter of train_test_split to '30' as that seemed to help increse the accuracy |
| Solution 3   | 0.7083333333333334 | [[102  21] [35  34]] | Used only 'glucose' feature and adjusted train_test_split parameter by changing random_state to "42" - Refer to lab3_version3.py for code |
| Baseline     | 0.6770833333333334 | [[114  16] [46  16]] | NA - Baseline Solution |