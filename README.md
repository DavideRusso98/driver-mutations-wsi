# Attention map
![Attentionmap on different patches](https://github.com/DavideRusso98/driver-mutations-wsi/blob/simple_training/attention_map_val.png?raw=true)

# Result
```
Testing BRCA1 over exprection

Accuracy: 0.6509

Precision: 0.6959

Recall: 0.8440

F1 Score: 0.7628

AUC-ROC: 0.6288

Log Loss: 0.6699

Matthews Correlation Coefficient (MCC): 0.1333

Confusion Matrix:
[[0.27 0.73]
 [0.16 0.84]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.46      0.27      0.34        71
         1.0       0.70      0.84      0.76       141

    accuracy                           0.65       212
   macro avg       0.58      0.56      0.55       212
weighted avg       0.62      0.65      0.62       212

    accuracy                           0.69       159
   macro avg       0.64      0.61      0.61       159
weighted avg       0.67      0.69      0.67       159
```
