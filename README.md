# Attention map
![Attentionmap on different patches](https://github.com/DavideRusso98/driver-mutations-wsi/blob/simple_training/attention_map_val.png?raw=true)

# Patch-level attention map
![Patch-level attention map](https://github.com/DavideRusso98/driver-mutations-wsi/blob/simple_training/old_attention_map_val.png?raw=true)

# Result
```
Testing BRCA1 over exprection

Accuracy: 0.6509

Precision: 0.7638

Recall: 0.6879

F1 Score: 0.7239

AUC-ROC: 0.6728

Log Loss: 0.7660

Matthews Correlation Coefficient (MCC): 0.2556

Confusion Matrix:
[[0.58 0.42]
 [0.31 0.69]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.48      0.58      0.53        71
         1.0       0.76      0.69      0.72       141

    accuracy                           0.65       212
   macro avg       0.62      0.63      0.62       212
weighted avg       0.67      0.65      0.66       212

```
