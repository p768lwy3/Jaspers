# Project of [Spooky Author Indetification](https://www.kaggle.com/c/spooky-author-identification)

## New:
  with embedding layers, it will have a much better result with a simple tokenizing and vectorizing.
  ![Accuracy](https://github.com/p768lwy3/Small_Project_Machine_Learning_for_selfstudy/blob/master/Kaggle/SpookyAuthorIdentification/pic/model_accuracy.png)
  ![Loss Function](https://github.com/p768lwy3/Small_Project_Machine_Learning_for_selfstudy/blob/master/Kaggle/SpookyAuthorIdentification/pic/model_loss.png)

## Old:
    1. Preprocessing ([data_utils.py](https://github.com/p768lwy3/Small_Project_Machine_Learning_for_selfstudy/blob/master/Kaggle/SpookyAuthorIdentification/data_utils.py)): </br>
        1. Tokenization</br>
        2. Throw away any words that occur too frequently or infrequently</br>
        3. Stemming words</br>
        4. Converting text into vector format</br>
    2. Classifier (main.py): </br>
        Build Neural Network for classifier</br>
    3. Record (model.txt): </br>
        Mark down the old version of neural network</br>
