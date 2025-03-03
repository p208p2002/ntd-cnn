# NTD-CNN
新台幣鈔票辨識
> 圖像識別課堂作業

## Methods
- Convolutional Neural Network
- Data Augmenting
- Dropout
- Early Stopping

## Dataset
100、500、1000元鈔票各20張
### Train setting
100、500、1000元鈔票各10張
### Test setting
100、500、1000元鈔票各10張

## Files
- `main.py`: 主程式
- `cnn_model.py`: CNN網路架構
- `core.py`: 自訂函式庫

## Results
### acc_score (training)
<img src="https://github.com/p208p2002/NTD-CNN/blob/master/acc_score.png?raw=true" alt="acc_score" width="380px"/>

### loss_score (training)
<img src="https://github.com/p208p2002/NTD-CNN/blob/master/loss_score.png?raw=true" alt="loss_score" width="380px"/>

### confusion_matrix (testing)
<img src="https://github.com/p208p2002/NTD-CNN/blob/master/confusion_matrix.png?raw=true" alt="confusion_matrix" width="320px"/>

## ENV
- torch >= 1.3.0
