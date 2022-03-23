import warnings
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

import statsmodels.api as sm

# import statsmodels.api as sm
# data = sm.datasets.co2.load_pandas().data
# print(data)
# exit()






def change_col_name(data, old_name, new_name):
    return data.rename(columns={old_name: new_name})


def load_data():
    
    data_orig = pd.read_csv('raw_data.csv')
    df = data_orig[["日期", "備轉容量(MW)"]]
    df = change_col_name(df, "日期", "Date")
    df = change_col_name(df, "備轉容量(MW)", "OR")
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')  # convert date column to DateTime
    df.set_index('Date', inplace=True)

    
    return df




def TestStationaryPlot(df):
    rol_mean = df.rolling(window = 1, center = False).mean()
    rol_std = df.rolling(window = 1, center = False).std()
    
    #plt.plot(figsize=(15, 8))
    plt.plot(df, color = 'blue',label = 'Original')
    plt.plot(rol_mean, color = 'red', linestyle='-.', label = 'Moving Average')
    plt.plot(rol_std, color ='black', linestyle='--', label = 'Standard Deviation')
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    plt.xlabel('Time', fontsize = 16)
    plt.ylabel('CO2', fontsize = 16)
    plt.legend(loc='best', fontsize = 16)
    plt.title('Moving Average and Standard Deviation', fontsize = 22)
    plt.show(block= True)

def TestStationaryAdfuller(df, cutoff = 0.2):
    from statsmodels.tsa.stattools import adfuller
    df_test = adfuller(df, autolag = 'AIC')
    df_test_output = pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in df_test[4].items():
        df_test_output['Critical Value (%s)'%key] = value
    print(df_test_output)

    if df_test[1] <= cutoff:
        print('拒绝原假设，即数据没有单位根,序列是平稳的。')
    else:
        print('不能拒绝原假设，即数据存在单位根,数据是非平稳序列。')
        
        



def deviate(df):
    from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
    from statsmodels.graphics.api import qqplot 
    y_diff = df.diff(1)
    y_seasonal_diff = y_diff - y_diff.shift(12)
    
    return y_seasonal_diff



def method2(y):
    #方法二：网格搜索
    import itertools
    # 首先定义 p、d、q 的参数值范围，这里取 0 - 2.
    p = d = q = range(0, 3)

    # 然后用itertools生成不同的参数组合
    pdq = list(itertools.product(p, d, q))
    
    # 同理处理季节周期性参数，也生成相应的多个组合
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    print(seasonal_pdq)
    print(pdq)
    exit()
    warnings.filterwarnings('ignore') 
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print('SARIMAX{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue




def white_noise(y_seasonal_diff):
    #再对1阶12步差分后序列做白噪声检验
    warnings.filterwarnings('ignore') 
    y_seasonal_diff.dropna(inplace = True)
    r,q,p = sm.tsa.acf(y_seasonal_diff.values.squeeze(), qstat=True) 
    data = np.c_[range(1,41), r[1:], q, p] 
    
    table = pd.DataFrame(data, columns=['lag', 'AC', 'Q', 'Prob(>Q)']) 
    print(table.set_index('lag')) 
    
    return




def train(y):
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()

    print(results.summary().tables[1])
    # results.plot_diagnostics(figsize=(12, 10))
    # plt.show()
    return results





def predict(results, date_start):
    pred = results.get_prediction(start=date_start, dynamic=False)
    pred_ci = pred.conf_int()
    
    return pred, pred_ci
    


def main():

    df = load_data()
    print(df)
    
    
    TestStationaryAdfuller(df)
    
    y_seasonal_diff = deviate(df)
    
    TestStationaryAdfuller(y_seasonal_diff.dropna(inplace=False))
    
    method2(df)
    exit()
    
    # ============= start training =============
    results = train(df)
    
    # ============= make prediction =============
    pred, pred_ci = predict(results, "20220101")
    
    
    ax = df.plot(label='Original')
    
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('OR')
    plt.legend()

    plt.show()






if __name__ == "__main__":
    main()
