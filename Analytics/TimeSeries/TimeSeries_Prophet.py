from fbprophet import Prophet
import CommonLib.Performance.PerformanceManager as pfm
import pandas as pd
import CommonLib.Common as common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import CommonLib.Report.WordManager as word

matplotlib.use('TkAgg')  # pyplot 시각화 에러나서 추가함.
common.DataFrame_PrintFull(pd)


def time_series_prophet_run(request_json, df_DB=None):
    """
    [2021.11.25 박성한]
    시계열 Prophet 모델 기능 정의

    :param df_DB:DB접속 넘겨받을 DataFrame.
    :param request_json: 요청 json 데이터
    :return:
    """
    try:
        outparam_trend = 'trend'
        outparam_yhat_lower = 'yhat_lower'
        outparam_yhat_upper = 'yhat_upper'
        outparam_trend_lower = 'trend_lower'
        outparam_trend_upper = 'trend_upper'
        outparam_predic_yhat = 'predic_yhat'
        outparam_predic_date = 'predic_date'

        col_name1 = 'ds'
        col_name2 = 'y'

        test_data_rate = 0.33
        number_cut = 3

        m = Prophet()
        m.daily_seasonality = True
        m.weekly_seasonality = True

        if __name__ == "__main__":
            df = pd.read_csv(common.get_local_file_path() + '시계열_CRM요청건수_2021_6월7월.csv')
        else:
            if df_DB is None:
                # CSV 파일 로드 형식
                df = pd.DataFrame(request_json)
            else:
                # DataBase로 부터 전달받은 DataFrame
                df = df_DB

        df = df.dropna()
        df.columns = [col_name1, col_name2]
        origin_df = df
        origin_df_copy = df.copy()
        origin_df_copy[col_name2] = origin_df_copy[col_name2].astype('int')  # 조건절 타입에 맞도록 변경

        # 훈련 데이터, 테스트 데이터 분할
        df[col_name2] = pd.to_numeric(df[col_name2])
        series = df[col_name2]
        x_value = series.values
        size = int(len(x_value) * test_data_rate)  # 테스트 데이터 33%

        test, train = df.iloc[0:size], df.iloc[size:len(df)]
        train[col_name1] = pd.to_datetime(train[col_name1])

        prophet = Prophet(seasonality_mode='multiplicative',
                          daily_seasonality=True,
                          changepoint_prior_scale=0.5)  # 기본 0.05 해당 수치가 높을 수록 추세가 더 유연해짐
        prophet.fit(train)

        # 날짜만 생성
        future_data = prophet.make_future_dataframe(periods=31)

        # trend, yhat_lower, yhat_upper, trend_lower, trend_upper
        forecast_data = prophet.predict(future_data)

        return_val = forecast_data[[col_name1, 'yhat',
                                    outparam_trend, outparam_yhat_lower, outparam_yhat_upper, outparam_trend_lower,
                                    outparam_trend_upper]]

        result_df = pd.DataFrame(return_val)
        result_df.columns = ['ds_result', 'yhat',
                             outparam_trend, outparam_yhat_lower, outparam_yhat_upper, outparam_trend_lower,
                             outparam_trend_upper]

        result_df[outparam_trend] = round(result_df[outparam_trend], number_cut)
        result_df[outparam_yhat_lower] = round(result_df[outparam_yhat_lower], number_cut)
        result_df[outparam_yhat_upper] = round(result_df[outparam_yhat_upper], number_cut)
        result_df[outparam_trend_lower] = round(result_df[outparam_trend_lower], number_cut)
        result_df[outparam_trend_upper] = round(result_df[outparam_trend_upper], number_cut)

        origin_df = origin_df.sort_values(by=col_name1, ascending=True)
        origin_df = origin_df.reset_index(drop=True)

        # min 값을 비교하기 위한 형변환
        result_df['ds_result'] = result_df['ds_result'].astype('str')

        # 모델결과 최소값 이하는 제외한다. (train, test)
        origin_df = origin_df[origin_df.ds >= result_df['ds_result'].min()]
        origin_df = origin_df.reset_index(drop=True)

        # 미래 예측값에 대한 결과값 추출
        predic_df = result_df[result_df.ds_result > origin_df[col_name1].max()]
        predic_df = predic_df.loc[:, ['ds_result', 'yhat']]

        predic_df.columns = [outparam_predic_date, outparam_predic_yhat]
        predic_df[outparam_predic_yhat] = round(predic_df[outparam_predic_yhat], number_cut)

        result_df = pd.concat([origin_df, result_df, predic_df], axis=1)
        result_df.reset_index(drop=True, inplace=True)

        # 정규분포 그래프
        histogram_df = pfm.make_histogram_data(origin_df)
        result_df = pd.concat([result_df, histogram_df], axis=1)

        # 기술통계 결과지표
        # sts_df = pfm.descriptive_statistics(np.array(origin_df_copy[col_name2]))
        sts_df = pfm.descriptive_statistics(np.array(result_df[outparam_predic_yhat].dropna()))
        result_df = pd.concat([result_df, sts_df], axis=1)

        # 결과 이미지 생성
        result_Image = common.Path_Prj_Main() + word.result_image_path + 'ProphetImg_' + common.Regexp_OnlyNumberbyDate() + '.png'
        font_size = word.result_font_size
        fig = plt.figure(figsize=(word.result_Img_width, word.result_Img_height))
        fig.set_facecolor('white')
        plt.plot(result_df['yhat'], color='red', marker='o', markersize=2, linestyle='dotted')  # 전체 미래치 Full 자료
        plt.plot(result_df['y'], color='#22a7f2', marker='o', markersize=2, linestyle='dashed')  # 원 데이터
        plt.xlabel(word.timeSeries_Xlabel, fontsize=font_size)
        plt.ylabel(word.timeSeries_Ylabel, fontsize=font_size)
        plt.axvspan(len(result_df['y']) - 31, len(result_df['y']), facecolor='#D5E9F6', edgecolor='grey', alpha=0.5, hatch='///')
        plt.savefig(result_Image)

        # 분석결과 리포트 생성
        report_fileName = word.CreateReport_Prophet(result_df, result_Image)
        fileNameArr = [report_fileName]
        result_df = pd.concat([result_df, pd.DataFrame(fileNameArr, columns=["FILENAME"])], axis=1)

        # 결과보고서 이미지변환
        # word.word_to_jpg(report_fileName)
        print(result_df.head())

        return result_df

    except Exception as err:
        common.exception_print(err)


if __name__ == "__main__":
    time_series_prophet_run(None)
