import CommonLib.Common as common
import CommonLib.Performance.PerformanceManager as pfm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import AgglomerativeClustering

common.DataFrame_PrintFull(pd)
range_tot_count = 8


def kmeans_run(request_json):
    """
    [2021.12.02 박성한]
    k-평균 군집화

    :param request_json: 요청 json데이터
    :return: 결과 DataFrame
    """
    try:
        col_name1 = 'x'
        col_name2 = 'y'
        col_name3 = 'label'
        col_input = 'Cluster_InputValue'

        n_clusters = 5
        result_arr = []

        if __name__ == "__main__":
            df = pd.read_csv(common.get_local_file_path() + '회귀_나이_진료시간.csv')
        else:
            df = pd.DataFrame(request_json)

        temp_df = df.copy()

        if len(temp_df.columns) == 3:
            input_df = temp_df.loc[:, col_input]
            input_df = input_df.dropna()
            input_df = input_df.reset_index(drop=True)
            n_clusters = int(input_df[0])

        df = df.iloc[:, [0, 1]]
        df.columns = [col_name1, col_name2]
        df = df.dropna()
        df[col_name1] = df[col_name1].astype('int')
        df[col_name2] = df[col_name2].astype('int')

        # 클러스터 갯수 체크
        check_cluster_df = check_cluster(df)
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10)
        y_pred = kmeans.fit_predict(df)

        # silhouette_score는 모든 샘플의 평균값을 제공
        # 이것은 형성된 클러스터의 밀도와 분리에 대한 관점 제공
        silhouette_avg = silhouette_score(df, y_pred)
        result_arr.append(str(silhouette_avg))

        # 각 샘플의 실루엣 점수 계산
        sample_silhouette_values = silhouette_samples(df, y_pred)
        result_arr.append(str(sample_silhouette_values))

        result_arr.append(kmeans.inertia_)
        result_arr.append(kmeans.labels_)
        result_arr.append(kmeans.algorithm)

        if __name__ == "__main__":
            # plt.scatter(df[:, 0], df[:, 1])
            plt.scatter(df.loc[:, col_name1], df.loc[:, col_name2])
            # plt.show()

            # print('='*100)
            # print(y_pred)

            # 각 라벨별로 색상 분리
            # plt.scatter(df[:, 0], df[:, 1], c=y_pred)
            plt.scatter(df.loc[:, col_name1], df.loc[:, col_name2], c=y_pred)
            # plt.show()

            # 점들의 중앙좌표를 표시
            plt.scatter(kmeans.cluster_centers_[:, 0],
                        kmeans.cluster_centers_[:, 1],
                        s=50, c='red')
            # plt.show()

        result_df = pd.concat([pd.DataFrame(df), pd.DataFrame(y_pred)], axis=1)
        result_df.columns = [col_name1, col_name2, col_name3]
        result_df = pd.concat([result_df, pd.DataFrame(result_arr)], axis=1)

        # 군집갯수 데이터
        result_df = pd.concat([result_df, check_cluster_df], axis=1)

        # 정규분포 그래프
        histogram_df = pfm.make_histogram_data(df)
        result_df = pd.concat([result_df, histogram_df], axis=1)

        # 기술통계 결과지표
        # make_histogram_data에 의해 colname 변경됨. copy() 메모리 관리상 사용하지 않고 진행.
        sts_df = pfm.descriptive_statistics(np.array(df['dependent']))
        result_df = pd.concat([result_df, sts_df], axis=1)

        # Score 추가
        score_df = hierarchical_clustering(df)
        result_df = pd.concat([result_df, score_df], axis=1)

        # 실루엣 Score 리턴
        silhouette_df = getlist_silhouette_score(df)
        result_df = pd.concat([result_df, silhouette_df], axis=1)

        # # 모델 성능지표
        if len(temp_df.columns) == 3:
            temp_df = temp_df.iloc[:, [0, 1]]

        # ols('반응변수 ~ 설명변수1+설명변수2+..', data=데이터).fit() (단순선형회귀모형 적합)
        filter_param = temp_df.columns[1] + ' ~ ' + temp_df.columns[0]
        temp_df = temp_df.dropna()
        temp_df[temp_df.columns[0]] = temp_df[temp_df.columns[0]].astype('int')
        temp_df[temp_df.columns[1]] = temp_df[temp_df.columns[1]].astype('int')
        temp_df = temp_df[temp_df != 0]

        fit = sm.OLS.from_formula(filter_param, data=temp_df).fit()
        result_func = [pfm.eval_all(temp_df[temp_df.columns[0]], fit.fittedvalues, True, fit)]
        result_func = pd.DataFrame(result_func, columns=['rst_radj', 'rst_mae', 'rst_mse', 'rst_rmse', 'rst_msle',
                                                         'rst_mpe', 'rst_mape', 'rst_me', 'rst_sse'])
        result_df = pd.concat([result_df, result_func], axis=1)

        # print(result_df.head())

        return result_df.transpose()

    except Exception as err:
        common.exception_print(err)


def check_cluster(check_df):
    """
    [2021.12.02 박성한]
    클러스터 갯수 체크

    :param check_df: 클러스터 갯수 체크대상 DataFrame
    :return: 결과 DataFrame
    """
    try:
        distortions = []
        for i in range(1, 9):
            km = KMeans(
                n_clusters=i, init='random',
                n_init=10, max_iter=300,
                tol=1e-04, random_state=0
            )
            km.fit(check_df)
            distortions.append(km.inertia_)  # 군집 내 분산, 적을수록 좋음

        df = pd.DataFrame(distortions, columns=['cluster_yval'])

        if __name__ == "__main__":
            # plot - 급격하게 줄어드는 부분
            plt.plot(range(1, 11), distortions, marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Distortion')
            plt.show()

        return df

    except Exception as err:
        common.exception_print(err)


def getlist_silhouette_score(df):
    """
    [2021.12.02 박성한]
    실루엣 점수 각 k별 점수 리턴
    :param df:
    :return:
    """
    try:
        k_range = range(2, range_tot_count)

        best_n = -1
        best_silhouette_score = -1

        result_arr_k = []
        result_arr_score = []
        result_arr_best = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10)
            colusters = kmeans.fit_predict(df)

            score = silhouette_score(df, colusters)

            result_arr_k.append(k)
            result_arr_score.append(score)

            if score > best_silhouette_score:
                best_n = k
                best_silhouette_score = score

        best_result = 'best K : ' + str(best_n) + '  score : ' + str(round(best_silhouette_score, 3))
        result_arr_best.append(best_result)

        k_df = pd.DataFrame(result_arr_k, columns=['K'])
        score_df = pd.DataFrame(result_arr_score, columns=['ScoreList'])
        best_df = pd.DataFrame(result_arr_best, columns=['BestScore'])

        return_df = pd.concat([k_df, score_df], axis=1)
        return_df = pd.concat([return_df, best_df], axis=1)

        # print(return_df.head())

        return return_df

    except Exception as err:
        common.exception_print(err)


def hierarchical_clustering(df):
    """
    [2021.12.02 박성한 추가]
    https://bcho.tistory.com/1204
    single_score : 두 클러스터내에서 가장 가까운 거리를 사용
    average_score : 각 클러스터내의 각 점에서 다른 클러스터내의 모든 점사이의 거리에 대한 평균을 사용
    complete_score : 두 클러스터상에서 가장 먼 거리를 이용해서 측정하는 방식
    :return:
    """
    try:
        linkages = ['single', 'average', 'complete']
        k_range = range(2, range_tot_count)
        k_silhouette_df = pd.DataFrame(k_range, columns=['k'])

        for connect in linkages:
            k_silhouette = []
            for k in k_range:
                clustering = AgglomerativeClustering(n_clusters=k, linkage=connect)
                clusters = clustering.fit_predict(df)
                score = silhouette_score(df, clusters)
                result = [score]
                k_silhouette.append(result)

            score_df = pd.DataFrame(k_silhouette, columns=[connect + '_score'])
            k_silhouette_df = pd.concat([k_silhouette_df, score_df], axis=1)

        return k_silhouette_df

    except Exception as err:
        common.exception_print(err)


if __name__ == "__main__":
    kmeans_run(None)
