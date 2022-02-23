import pandas as pd
import CommonLib.Common as common
import statsmodels.api as sm
import CommonLib.Performance.PerformanceManager as pfm
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

common.DataFrame_PrintFull(pd)


def MultipleLinearRegression(request_json):

    try:

        if __name__ == "__main__":
            df = pd.read_csv(common.get_local_file_path() + '다중회귀_혈압나이성별온도.csv')
        else:
            df = pd.DataFrame(request_json)

        df = df.dropna(axis=0)
        df = df.astype('float')
        corr_df = df.copy()

        # 다중회귀분석
        target = df.iloc[:, [0]]  # 종속변수
        x_data = df.iloc[:, 1:]   # 독립변수들
        x_data1 = sm.add_constant(x_data, has_constant="add")  # 상수항 컬럼추가

        multi_model = sm.OLS(target, x_data1)
        ols_regression = multi_model.fit()

        # 모델 성능지표 target:원본자료, ols_regression.fittedvalues:예측결과치
        model_result = pfm.eval_all(target, ols_regression.fittedvalues)
        model_df = pd.DataFrame(model_result, columns=['P_MODEL_RESULT'])

        # SUMMARY > DATAFRAME
        summary_arr = [str(ols_regression.summary())]
        df_summary = pd.DataFrame(summary_arr)
        df_summary.rename(columns={df_summary.columns[0]: "SUMMARY"}, inplace=True)

        # 다중공선성 확인 (VIF)
        '''상관관계는 -1~1의 분포를 지니는데 여기서 0.5가 넘어가는 
        변수들간의 상관관계가 빈출되는 것은 충분히 다중공선성 발생을 의심
        보통은 VIF가 10보다 크면 다중공선성이 있다고 판단'''
        '''독립변수들간의 강한 상관관계가 있으면 한가지는 배제 해야한다는 것'''

        x_data = x_data.astype('float')
        vif = pd.DataFrame()

        # variance_inflation_factor 함수설명 (참고 블로그 : https://sosoeasy.tistory.com/386)
        # 독립변수간에 상관관계가 높을 수록 과적합되거나 정확한 분석이 되지 않을 수 있음.
        # -> 상관성을 확인하고 상관이 있는 변수들은 제거 (VIF-분산 인플레이션 계수:Variance Inflation Factor 로 변수를 제거)
        vif["VIFFactor"] = [variance_inflation_factor(x_data.values, i) for i in range(x_data.shape[1])]
        vif["features"] = x_data.columns
        vif.VIFFactor = vif.VIFFactor.round(3)  # 소숫점 3자리

        vif = vif.sort_values(by=vif.columns[0], ascending=False)
        vif = vif.reset_index(drop=True)

        # 상관관계 메트릭스
        corr_df2 = pfm.Correlation(corr_df)
        result_all = pd.concat([df_summary, vif], axis=1)
        result_all = pd.concat([result_all, pd.DataFrame(corr_df2)], axis=1)

        # 선형그래프
        x_train, x_test, y_train, y_test = train_test_split(x_data, target, train_size=0.8, test_size=0.2)
        linear = LinearRegression()
        linear.fit(x_train, y_train)
        y_predict = linear.predict(x_test)  # 예측 종속변수
        y_test = y_test.reset_index(drop=True)
        y_test.columns = ['x']

        # 정규분포
        indepen = df.iloc[:, [1]]
        origin_df = pd.concat([indepen, target], axis=1)
        origin_df.columns = ['col_indepen', 'col_depen']
        origin_df_copy = origin_df.copy()
        histogram_df = pfm.make_histogram_data(origin_df)

        # 기술통계 지표
        origin_df_copy = origin_df_copy.dropna()
        origin_df_copy = origin_df_copy.astype('int')
        sts_df = pfm.descriptive_statistics(np.array(origin_df_copy['col_depen']))

        # 실제 종속변수(y_test)와 예측 종속변수(y_predict)
        result_all = pd.concat([result_all, y_test], axis=1)
        result_all = pd.concat([result_all, pd.DataFrame(y_predict, columns=['y'])], axis=1)

        # 정규분포
        result_all = pd.concat([result_all, histogram_df], axis=1)

        # 기술통계
        result_all = pd.concat([result_all, sts_df], axis=1)

        # 모델 성능지표
        result_all = pd.concat([result_all, model_df], axis=1)

        return result_all.transpose()

    except Exception as err:
        common.exception_print(err)


if __name__ == "__main__":
    MultipleLinearRegression(None)
