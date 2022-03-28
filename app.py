import datetime
import pandas as pd
from SARIMA import SARIMA




def write_output(df, output_file):
    base = datetime.date.fromisoformat('2022-03-30')

    date_list = []
    for x in range(15):
        date = base + datetime.timedelta(days=x) 
        date_list.append(date.strftime('%Y%m%d'))

    data_list = []
    for data in df:
        data_list.append(data)

    output = pd.DataFrame({'date':date_list, 'operating_reserve(MW)':data_list}) 
    output.to_csv(output_file, index = False)
    
    return  




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    model = SARIMA()
    prediction_result = model.main(args.training)
    write_output(prediction_result, args.output)

