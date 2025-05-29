# Animal Image Classification

This repository contains the code and data for an image classification task to identify animals among three classes: **dog**, **chicken**, and **spider**. The model was trained and evaluated using a custom dataset, with training performed on an NVIDIA H100 80GB GPU.

---

## Dataset Structure

The dataset is organized into training, validation, and test splits, each containing three categories of animal images:

```
./data/train/
├── dog/       # 3,890 images
├── chicken/   # 2,478 images
└── spider/    # 3,856 images

./data/val/
├── dog/       # 486 images
├── chicken/   # 309 images
└── spider/    # 482 images

./data/test/
├── dog/       # 487 images
├── chicken/   # 311 images
└── spider/    # 483 images
```

**Total images:**

- Training: 10,224 images  
- Validation: 1,277 images  
- Test: 1,281 images

---

## Model Training

- Training hardware: **NVIDIA H100 80GB HBM3**
- The model architecture and training procedure are implemented in `notebook.ipynb`.
- The dataset is preprocessed and loaded from the folder structure shown above.

---

## Results on Test Set

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Chicken  | 0.93      | 0.92   | 0.93     | 311     |
| Dog      | 0.96      | 0.93   | 0.94     | 487     |
| Spider   | 0.95      | 0.97   | 0.96     | 483     |

**Overall Accuracy:** 95%  
**Macro Avg:** Precision 0.94, Recall 0.94, F1-score 0.94  
**Weighted Avg:** Precision 0.95, Recall 0.95, F1-score 0.95

---

## How to Run

1. Clone the repository.

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook to train and evaluate the model:
   ```bash
   jupyter notebook notebook.ipynb
   ```

---

## Notes

- Ensure the dataset is placed in the `./data/` directory following the described folder structure.
- Adjust paths or hyperparameters in the notebook if necessary.

---

Feel free to reach out for questions or suggestions!
