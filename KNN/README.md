# K-Nearest Neighbours (KNN)

Simple KNN classification using **scikit-learn** on the Iris dataset.

## Run it

```bash
pip install scikit-learn pandas matplotlib
```

Then open `knn.ipynb` and run the cells top to bottom.

## What the notebook does

| Step | What happens |
|------|--------------|
| 1 | Load the Iris dataset (150 flowers, 4 features, 3 species) |
| 2 | Split into 80% train / 20% test |
| 3 | Scale the features with `StandardScaler` |
| 4 | Train `KNeighborsClassifier(n_neighbors=5)` |
| 5 | Check accuracy + confusion matrix |
| 6 | Try k = 1..20 and plot which k works best |
| 7 | Predict a brand new flower and show its 5 neighbours |

## Key points to remember

- KNN does **no real training** — it stores the data and only does the work at prediction time (*lazy learner*).
- It classifies by **majority vote** of the `k` nearest points, using Euclidean distance.
- **Always scale your features.** KNN is distance-based, so a large-range column would otherwise dominate.
- Fit the scaler on the **training set only**, then apply it to the test set — otherwise you leak test information.
- Small `k` = overfits noise, large `k` = too smooth. Tune it (properly, with cross-validation).

Result on this dataset: **93.3% accuracy** with k = 5.
