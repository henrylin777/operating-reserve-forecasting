# operating-reserve-forecasting

Operating-reserve load forecasting via Seasonal-ARIMA (SARIMA)

這次的任務我選用 statistical-based model 來進行備轉容量(OR)預測而非 Deep Learning based model
原因是因為 DL-based model 需要大量的訓練資料才能達到較佳的效果，考量到目前只有找到2020年~2022年3月的資料，其資料筆數大約為1千多筆，資料量顯然不夠。

### Usage

```
python3 app.py --training raw_data.csv --outout submission.csv 
```

### SARIMA 

Autoregressive Integrated Moving Average(ARMA)是以Autoregressive model(AR)和Moving Average(MA)這兩個model為基礎所構成。
但ARIMA並沒有將季節的因素考慮進去，於是就有了SARIMA的出現。

SARIMA的參數有 (p, q, d) x (P, Q, D, m)，我使用 grid method 搭配 Akaike Information Criterion (AIC)來找出最合適的參數，
找到之後就可以使用模型來進行預測。

#### training process

1. 根據時間序列的散點圖、自我相關函數和偏自我相關函數圖識別其平穩性。
2. 對非平穩的時間序列數據進行平穩化處理。直到處理後的自我相關函數和偏自我相關函數的數值非顯著非零。
3. 根據所識別出來的特徵建立相應的時間序列模型。平穩化處理後，若偏自我相關函數是截尾的，而自我相關函數是拖尾的，則建立AR模型；若偏自我相關函數是拖尾的，而自我相關函數是截尾的，則建立MA模型；若偏自我相關函數和自我相關函數均是拖尾的，則序列適合ARMA模型。
4. 參數估計，檢定是否具有統計意義。
5. 假說檢定，判斷（診斷）殘差序列是否為白噪聲序列。
6. 利用已通過檢定的模型進行預測。
