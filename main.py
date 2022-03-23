import csv 
import re 
import pandas as pd 
import matplotlib.pyplot as plt 



def transform_data(raw_data, column, out_type="str"):

    temp = [ None for i in range(len(raw_data[column]))]

    for i in range(len(raw_data[column])):
        temp[i] = str(raw_data[column][i])
    
    raw_data[column] = temp

    return raw_data





def make_figure(raw_data):

    print(raw_data["備轉容量(MW)"])

    plt.style.use("ggplot")# 使用ggplot主題樣式

    #畫第一條線，plt.plot(x, y, c)參數分別為x軸資料、y軸資料及線顏色 = 紅色
    plt.plot(raw_data["日期"], raw_data["備轉容量(MW)"], c = "r")  
    #畫第二條線，plt.plot(x, y, c)參數分別為x軸資料、y軸資料、線顏色 = 綠色及線型式 = -.
    # plt.plot(motor["YearMonth"], motor["Electric(thousand)"], "g-.")

    # 設定圖例，參數為標籤、位置
    plt.legend(labels=["Operating Reserve"], loc = 'best')
    plt.xlabel("YearDate", fontweight = "bold")                # 設定x軸標題及粗體
    plt.ylabel("Operating Reserve (MW)", fontweight = "bold")    # 設定y軸標題及粗體
    plt.title("This is topic", fontsize = 15, fontweight = "bold", y = 1.1)   # 設定標題、文字大小、粗體及位置
    plt.xticks(rotation=45)   # 將x軸數字旋轉45度，避免文字重疊

    plt.savefig("Motorcycles growth.jpg", bbox_inches='tight', pad_inches=0.0)     


    plt.show()
    plt.close()


def read_csv(csvFile):
    with open(csvFile, newline='') as infile:
        rows = csv.reader(infile)

        for row in rows:
            print(row[3])



def main():
    # read_csv("raw_data.csv")
    raw_data = pd.read_csv("raw_data.csv")
    raw_data = transform_data(raw_data, "日期", "str")

    print(raw_data.head(3))
    print(len(raw_data["備轉容量(MW)"]))
    # make_figure(raw_data)

if __name__ == "__main__":
    main()
