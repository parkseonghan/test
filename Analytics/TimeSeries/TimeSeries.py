import numpy as np
import pandas as pd

import CommonLib.Performance.PerformanceManager as Pfm
import CommonLib.Common as common
import CommonLib.Report.WordManager as word

# from statsmodels.tsa.arima_model import ARIMA -> 아래와 같이 변경
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot

common.DataFrame_PrintFull(pd, 500, 100)

# 지수표현 변환
pd.options.display.float_format = '{:.5f}'.format


def time_series_run(request_json):
    """
    [2021.11.25 박성한]
    시계열 ARIMA 모델 기능 정의
    :param request_json:요청 json
    :return:
    """
    try:

        # CSV 파일 로드시 일자, 금액 순서.
        col_name1 = 'DATE'
        col_name2 = 'AMT'
        col_maxcnt = 5

        # 결과리턴 배열변수
        result = []

        if __name__ == "__main__":
            df = pd.read_csv(common.get_local_file_path() + '시계열_CRM요청건수_2021_6월7월.csv')
        else:
            df = pd.DataFrame(request_json)

        column_count = len(df.columns)

        # ARIMA 차수 수동입력의 경우
        if column_count == col_maxcnt:
            df_arima = df.loc[:, ['p', 'd', 'q']]
            df_arima = df_arima.dropna()  # 결측치 제거
            df = df.iloc[:, [0, 1]]  # Date, Amt
            auto_yn = True
        else:
            auto_yn = False

        df = df.dropna()
        df_auto_arima = getresult_auto_arima(df, col_name1, col_name2, auto_yn)
        df.columns = [col_name1, col_name2]
        origin_df = df
        origin_df_copy = df.copy()
        origin_df_copy[col_name2] = origin_df_copy[col_name2].astype('int')  # 조건절 타입에 맞도록 변경

        # 금액 컬럼 타입 변경 Object -> float
        df[col_name2] = pd.to_numeric(df[col_name2])

        series = df[col_name2]

        x_value = series.values
        x_value = np.nan_to_num(x_value)  # 널savefig값을 0으로 대체

        # size = int(len(x_value) * 0.66)  # 훈련데이터 66%
        size = int(len(x_value) * 0.30)  # 훈련데이터 30%

        train, test = x_value[0:size], x_value[size:len(x_value)]  # train과 test로 데이터셋 분리
        test = x_value[size:len(x_value)]

        history = [index for index in train]
        predictions = list()  # 예측변수
        result_log = list()   # 결과출력 로그

        for t in range(len(test)):
            # p:자기회귀 차수, d:차분 차수, q:이동평균 차수
            if column_count == col_maxcnt:
                model = ARIMA(history, order=(int(df_arima.at[df_arima.index[-1], 'p']),
                                              int(df_arima.at[df_arima.index[-1], 'd']),
                                              int(df_arima.at[df_arima.index[-1], 'q'])))
            else:
                model = ARIMA(history, order=(1, 1, 0))

            # fit() 모델에 데이터를 적용하여 훈련시킴
            model_fit = model.fit()

            # forecast 예측 수행
            output = model_fit.forecast()

            # 모델 출력 결과를 yhat에 저장
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)

            # 모델 실행 결과를 Predicted로 출력해서 test로 분리해 둔 데이터를 expected로 사용하여 출력
            print('PredicData:' + str(round(yhat, 3)) + ',Original:' + str(obs))  # 로그 임시주석
            result_log.append('PredicData:' + str(round(yhat, 3)) + ',Original:' + str(obs))

        # 손실 함수로 평균제곱 오차 사용
        if __name__ == "__main__":
            result_Image = common.Path_Prj_Main() + word.result_image_path + 'ARIMAImg_' + common.Regexp_OnlyNumberbyDate() + '.png'
            font_size = word.result_font_size
            fig = pyplot.figure(figsize=(word.result_Img_width, word.result_Img_height))
            fig.set_facecolor('white')
            pyplot.plot(predictions, color='red', marker='o', markersize=2, linestyle='dotted')  # 전체 미래치 Full 자료
            pyplot.plot(test, color='#22a7f2', marker='o', markersize=2, linestyle='dashed')  # 원 데이터
            pyplot.xlabel(word.timeSeries_Xlabel, fontsize=font_size)
            pyplot.ylabel(word.timeSeries_Ylabel, fontsize=font_size)
            pyplot.savefig(result_Image)

        summary_arr = [str(model_fit.summary())]
        df_summary = pd.DataFrame(summary_arr)
        df_summary.rename(columns={df_summary.columns[0]: "SUMMARY"}, inplace=True)

        result.append(test)
        result.append(predictions)
        result.append(result_log)

        eval_test = np.array(test)
        eval_predic = np.array(predictions)

        # 0이 들어간 데이터는 제외
        eval_test = eval_test[eval_test != 0]

        # 배열의 사이즈를 동일하게 처리하기 위해 마지막 인덱스 조정
        if eval_test.size > eval_predic.size:
            eval_test = np.delete(eval_test, eval_test.size - 1)
        elif eval_predic.size > eval_test.size:
            eval_predic = np.delete(eval_predic, eval_predic.size - 1)

        result.append(Pfm.eval_all(eval_test, eval_predic))
        result.append(Pfm.get_clf_eval(eval_test, eval_predic))

        result_df = pd.DataFrame(result)
        result_df.reset_index(drop=True, inplace=True)
        result_df = result_df.transpose()
        result_df = pd.concat([result_df,
                               df_auto_arima,
                               pd.DataFrame(df_summary)], axis=1)

        # 정규분포 그래프
        histogram_df = Pfm.make_histogram_data(origin_df)
        result_df = pd.concat([result_df, histogram_df], axis=1)

        # 기술통계 결과지표
        sts_df = Pfm.descriptive_statistics(np.array(origin_df_copy[col_name2]))
        result_df = pd.concat([result_df, sts_df], axis=1)

        # 분석결과 리포트 생성
        # fileNameArr = [word.CreateReport_ARIMA(result_df, result_Image)]
        # result_df = pd.concat([result_df, pd.DataFrame(fileNameArr, columns=["FILENAME"])], axis=1)

        return result_df

    except Exception as err:
        common.exception_print(err)


def time_series_show_data(request_json):
    try:
        # CSV 파일 로드시 일자, 금액 순서.
        col_name1 = 'DATE'
        col_name2 = 'AMT'

        if __name__ == "__main__":
            df = pd.read_csv(common.get_local_file_path() + 'TimeSeries_small.csv')
        else:
            df = pd.DataFrame(request_json)

        df.reset_index(drop=True, inplace=True)

        df = df.dropna(axis=0)
        df.columns = [col_name1, col_name2]
        df[col_name1] = pd.to_datetime(df[col_name1])
        df[col_name2] = df[col_name2].astype(float)

        model = ARIMA(df[col_name2], order=(1, 1, 0))
        model_fit = model.fit()

        # 오차
        residuals = pd.DataFrame(model_fit.resid)
        residuals.plot()
        pyplot.show()
        residuals.plot(kind='kde')
        pyplot.show()
        return df

    except Exception as err:
        common.exception_print(err)


def getresult_auto_arima(df, col_name1, col_name2, auto_yn):
    try:
        order = [3, 3, 3]

        if auto_yn:
            df.columns = [col_name1, col_name2, 'p', 'd', 'q']
        else:
            df.columns = [col_name1, col_name2]

        # 금액 컬럼 타입 변경 Object -> float
        df[col_name2] = pd.to_numeric(df[col_name2])
        series = df[col_name2]

        x_value = series.values

        # train과 test로 데이터셋 분리
        size = int(len(x_value) * 0.66)
        test, train = x_value[0:size], x_value[size:len(x_value)]

        history = [index for index in train]

        p_vals = []
        d_vals = []
        q_vals = []
        aic_vals = []
        tot_vals = []

        for p in range(order[0]):
            for d in range(order[1]):
                for q in range(order[2]):
                    try:
                        model = ARIMA(history, order=(p, d, q))
                        model_fit = model.fit()

                        if not np.isnan(model_fit.aic):
                            p_order = p
                            d_order = d
                            q_order = q
                            aic = round(model_fit.aic, 3)

                            p_vals.append(p_order)
                            d_vals.append(d_order)
                            q_vals.append(q_order)
                            aic_vals.append(aic)

                            total_vals = [p_order, d_order, q_order, aic]
                            tot_vals.append(total_vals)

                    except Exception as err:
                        common.exception_print(err)

                    result_df = pd.DataFrame(list(zip(p_vals, d_vals, q_vals, aic_vals, tot_vals)),
                                             columns=['p', 'd', 'q', 'AIC', 'TOT'])

                    result_df.sort_values(by=['AIC'], inplace=True, ascending=True)
                    result_df.reset_index(drop=True, inplace=True)

        return result_df

    except Exception as err:
        common.exception_print(err)


def get_modelfit(model_fit):
    try:
        cut_number = 3
        result = [round(model_fit.aic, cut_number),
                  round(model_fit.bic, cut_number),
                  round(model_fit.hqic, cut_number),
                  round(model_fit.llf, cut_number),
                  round(model_fit.loglikelihood_burn, cut_number),
                  round(model_fit.mae, cut_number),
                  round(model_fit.mse, cut_number),
                  round(model_fit.sse, cut_number)]

        return pd.DataFrame(result)

    except Exception as err:
        common.exception_print(err)


if __name__ == "__main__":
    time_series_run(None)
