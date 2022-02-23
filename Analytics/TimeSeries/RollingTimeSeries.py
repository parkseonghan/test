import pandas as pd
import matplotlib.pyplot as plt

import CommonLib.Common as common
import CommonLib.Performance.PerformanceManager as pfm


def rolling_time_series_run(request_json, df_DB=None):
    """
    [2021.11.25 박성한]
    이동평균 그래프 기능 정의

    :param df_DB:
    :param request_json:요청 json
    :return:
    """
    try:
        if __name__ == "__main__":
            df = pd.read_csv(common.get_local_file_path() + '시계열_2개월_외래내원_일별환자수_201803_201804.csv')
        else:
            if df_DB is None:
                # CSV 파일 로드 형식
                df = pd.DataFrame(request_json)
            else:
                # DataBase로 부터 전달받은 DataFrame
                df = df_DB

        df = df.dropna()

        col_name1 = 'DATE'
        col_name2 = 'AMT'

        df.columns = [col_name1, col_name2]
        origin_df = df

        # 이동평균 : window는 단위 - 5일 단위, 10일 단위, 20일 단위
        roll_mean5 = pd.Series.rolling(df[col_name2], window=5, center=False).mean()
        roll_mean10 = pd.Series.rolling(df[col_name2], window=10, center=False).mean()
        roll_mean20 = pd.Series.rolling(df[col_name2], window=20, center=False).mean()

        if __name__ == "__main__":
            fig = plt.figure(figsize=(12, 4))
            chart = fig.add_subplot(1, 1, 1)

            chart.plot(df[col_name2], color='blue', label='High Column')
            chart.plot(roll_mean5, color='red', label='5 Day Rolling Mean')
            chart.plot(roll_mean10, color='orange', label='10 Day Rolling Mean')
            chart.plot(roll_mean20, color='pink', label='20 Day Rolling Mean')

            plt.legend(loc='best')
            plt.xlabel('Request Date', fontsize=15)
            plt.show()

        result = [df[col_name2], roll_mean5, roll_mean10, roll_mean20]

        result_data = pd.DataFrame(result)
        result_data.reset_index(drop=True, inplace=True)
        result_data = result_data.transpose()

        histogram_df = pfm.make_histogram_data(origin_df, 17)
        result_data = pd.concat([result_data, histogram_df], axis=1)

        return result_data

    except Exception as err:
        common.exception_print(err)


if __name__ == '__main__':
    rolling_time_series_run(None)
