# Attention map(not updated)
![Attentionmap on different patches](https://github.com/DavideRusso98/driver-mutations-wsi/blob/simple_training/attention_map_val.png?raw=true)

# Patch-level attention map(not updated)
![Patch-level attention map](https://github.com/DavideRusso98/driver-mutations-wsi/blob/simple_training/old_attention_map_val.png?raw=true)

# Result (DS_ABMIL)
```
Testing BRCA1 over exprection

######## Testing model 1 ########

Accuracy: 0.6981

Precision: 0.7305

Recall: 0.8652

F1 Score: 0.7922

AUC-ROC: 0.6762

Log Loss: 0.6693

Matthews Correlation Coefficient (MCC): 0.2671

Confusion Matrix:
[[0.37 0.63]
 [0.13 0.87]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.58      0.37      0.45        71
         1.0       0.73      0.87      0.79       141

    accuracy                           0.70       212
   macro avg       0.65      0.62      0.62       212
weighted avg       0.68      0.70      0.68       212


######## Testing model 2 ########

Accuracy: 0.6085

Precision: 0.6959

Recall: 0.7305

F1 Score: 0.7128

AUC-ROC: 0.5968

Log Loss: 0.9601

Matthews Correlation Coefficient (MCC): 0.0994

Confusion Matrix:
[[0.37 0.63]
 [0.27 0.73]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.41      0.37      0.39        71
         1.0       0.70      0.73      0.71       141

    accuracy                           0.61       212
   macro avg       0.55      0.55      0.55       212
weighted avg       0.60      0.61      0.60       212


######## Testing model 3 ########

Accuracy: 0.6321

Precision: 0.7299

Recall: 0.7092

F1 Score: 0.7194

AUC-ROC: 0.6161

Log Loss: 0.9471

Matthews Correlation Coefficient (MCC): 0.1857

Confusion Matrix:
[[0.48 0.52]
 [0.29 0.71]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.45      0.48      0.47        71
         1.0       0.73      0.71      0.72       141

    accuracy                           0.63       212
   macro avg       0.59      0.59      0.59       212
weighted avg       0.64      0.63      0.63       212


######## Testing model 4 ########

Accuracy: 0.6274

Precision: 0.7095

Recall: 0.7447

F1 Score: 0.7266

AUC-ROC: 0.6071

Log Loss: 0.9585

Matthews Correlation Coefficient (MCC): 0.1429

Confusion Matrix:
[[0.39 0.61]
 [0.26 0.74]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.44      0.39      0.41        71
         1.0       0.71      0.74      0.73       141

    accuracy                           0.63       212
   macro avg       0.57      0.57      0.57       212
weighted avg       0.62      0.63      0.62       212


######## Testing model 5 ########

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


########################################

############# Final Report #############

########################################

Accuracy: 0.6434

Precision: 0.7259

Recall: 0.7475

F1: 0.7350

Auc_roc: 0.6338

Log_loss: 0.8602

Mcc: 0.1901

Confusion Matrix:
[[0.44 0.56]
 [0.25 0.75]]

Best model for accuracy: 1 with 0.6981

Best model for f1: 1 with 0.7922

Best model for auc_roc: 1 with 0.6762

```
