from typing import Hashable
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import CommonLib.Common as common
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import CommonLib.Performance.PerformanceManager as pfm
import statsmodels.api as sm
# DBScan
from sklearn.cluster import DBSCAN


common.DataFrame_PrintFull(pd)


def DBscan_run(request_json):

    try:
        col_name1 = 'x'
        col_name2 = 'y'
        col_name3 = 'group'
        input_val = 0.5

        if __name__ == "__main__":
            df = pd.read_csv(common.get_local_file_path() + "DBScan_사망환자_나이_재원일수.csv")
        else:
            df = pd.DataFrame(request_json)

        if len(df.columns) == 4:
            df_input_independent = df.loc[:, 'eps_InputValue']
            df_input_independent = df_input_independent.dropna()
            df_input_independent = df_input_independent.reset_index(drop=True)
            input_val = df_input_independent[0]
            df = df.iloc[:, [0, 1, 2]]

        df = df.dropna()
        df.columns = [col_name1, col_name2, col_name3]
        df_result = df.iloc[:, len(df.columns) - 1]
        df[col_name3] = df_result
        origin_df = df.copy()

        eps_val = input_val
        min_samples_val = 5

        if __name__ == "__main__":
            fkg, ax = plt.subplots()
            colors = {1: 'red', 0: 'blue'}

            grouped = df.groupby(col_name3)
            key: Hashable
            for key, group in grouped:
                group.plot(ax=ax, kind='scatter', x=col_name1, y=col_name2, label=key, color=colors[key])
            plt.show()

        # 표준화 (평균=0, 분산=1)
        scale = StandardScaler()
        scale.fit(df[[col_name1, col_name2]])
        scaled_x = scale.transform(df[[col_name1, col_name2]])

        df['scaled_x'] = scaled_x[:, 0]
        df['scaled_y'] = scaled_x[:, 1]
        df.loc[:, ['scaled_x', 'scaled_y']] = df.loc[:, ['scaled_x', 'scaled_y']].apply(lambda x: round(x, 3))

        # k-means와 다르게 군집수를 정하지 않음. 자동으로 최적의 군집수를 찾아가는 알고리즘.
        # eps: 한 데이터가 주변에 얼만큼 떨어진 거리를 같은 군집으로 생각할지의 기준거리 (데이터 포인트 중심으로 반지름을 뜻함)
        # min_samples: 적어도 한군집에는 5개 이상 샘플이 모여야 군집으로 인정함
        dbscan = DBSCAN(eps=float(eps_val), min_samples=min_samples_val)  # 기본값.
        # dbscan = DBSCAN(eps=0.3, min_samples=min_samples_val)  # 기본값.
        cluster = dbscan.fit_predict(scaled_x)
        df['cluster'] = cluster

        if __name__ == "__main__":
            plt.scatter(x=df.scaled_x, y=df.scaled_y, c=df.cluster)
            plt.title("DBScan")
            plt.xlabel(col_name1)
            plt.ylabel(col_name2)
            plt.show()

        result_df = pd.concat([df.scaled_x, df.scaled_y], axis=1)   # DBScan 좌표 결과값
        result_df = pd.concat([result_df, df.cluster], axis=1)      # DBScan 그룹화 결과값
        result_df = pd.concat([result_df, origin_df], axis=1)       # 선택된 원본 자료

        # 모델 성능지표
        df = df.loc[:, ['x', 'y']]
        ols_param = df.columns[1] + ' ~ ' + df.columns[0]
        df[df.columns[0]] = df[df.columns[0]].astype('int')
        df[df.columns[1]] = df[df.columns[1]].astype('int')
        df = df[df > 0]
        df = df.dropna()
        df = df.reset_index(drop=True)

        fit = sm.OLS.from_formula(ols_param, data=df).fit()
        result_func = [pfm.eval_all(df[df.columns[0]], fit.fittedvalues, True, fit)]
        result_func = pd.DataFrame(result_func, columns=['rst_radj', 'rst_mae', 'rst_mse', 'rst_rmse', 'rst_msle',
                                                         'rst_mpe', 'rst_mape', 'rst_me', 'rst_sse'])
        result_df = pd.concat([result_df, result_func], axis=1)

        # 기술통계 지표
        sts_df = pfm.descriptive_statistics(np.array(df[df.columns[1]]))
        result_df = pd.concat([result_df, sts_df], axis=1)

        # 정규분포 그래프
        df = df.iloc[:, [0, 1]]
        histogram_df = pfm.make_histogram_data(df)
        result_df = pd.concat([result_df, histogram_df], axis=1)

        result_arr = checkDBSCAN(df, eps=0.2, min_samples=5)
        result_arr += checkDBSCAN(df, eps=0.3, min_samples=5)
        result_arr += checkDBSCAN(df, eps=0.4, min_samples=5)
        result_arr += checkDBSCAN(df, eps=0.5, min_samples=5)
        result_arr += checkDBSCAN(df, eps=0.6, min_samples=5)
        result_arr += checkDBSCAN(df, eps=0.7, min_samples=5)
        result_arr += checkDBSCAN(df, eps=0.8, min_samples=5)

        summary_arr = [result_arr]
        result_df = pd.concat([result_df, pd.DataFrame(summary_arr, columns=["SUMMARY"])], axis=1)
        # print(result_df.head())
        return result_df.transpose()

    except Exception as err:
        common.exception_print(err)


def checkDBSCAN(data, eps, min_samples):
    try:
        # create model and prediction
        data = data.iloc[:, [0, 1]]
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        scale = StandardScaler()
        scale.fit(data)
        scaled_x = scale.transform(data)
        db.fit_predict(scaled_x)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        string_gubun = '==========================================' + "\n"
        string_gubun += f'*** eps={eps}, min_samples={min_samples} ***' + "\n"
        string_gubun += 'Estimated number of clusters: %d' % n_clusters_ + "\n"
        string_gubun += 'Estimated number of noise points: %d' % n_noise_ + "\n"
        string_gubun += "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, labels) + "\n"

        return string_gubun

    except Exception as err:
        common.exception_print(err)


if __name__ == "__main__":
    DBscan_run(None)
