# Intercepting driver mutations in WSI

One of the most challenging aspects in pathology is identifying cancer-relevant driver mutations within WSIs. 
This project aims to train a model to distinguish between breast cancer patients with and without the BRCA over-expression.



```bash
./gdc-client download -m [manifest_name.txt]
```

# Best result
```
Accuracy: 0.6918

Precision: 0.7280

Recall: 0.8585

F1 Score: 0.7879

AUC-ROC: 0.6737

Log Loss: 0.6360

Matthews Correlation Coefficient (MCC): 0.2495

Confusion Matrix:
[[0.36 0.64]
 [0.14 0.86]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.56      0.36      0.44        53
         1.0       0.73      0.86      0.79       106

    accuracy                           0.69       159
   macro avg       0.64      0.61      0.61       159
weighted avg       0.67      0.69      0.67       159
```
