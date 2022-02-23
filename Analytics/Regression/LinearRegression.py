import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import CommonLib.Common as common
import CommonLib.Performance.PerformanceManager as pfm
import CommonLib.Report.WordManager as word
import statsmodels.api as sm
import numpy as np

matplotlib.use('TkAgg')  # pyplot 시각화 에러나서 추가함.


def linear_regression_run(request_json):
    """
    [2021.11.25 박성한]
    단일 회귀분석(OLS) 기능정의

    :param request_json: 요청 json
    :return:
    """
    try:
        # independent  독립변수 X
        # dependent    종속변수 Y

        # CSV 파일 로드시 일자, 금액 순서
        col_indepen = 'INDEPEN'
        col_depen = 'DEPEN'
        result_return_value = []

        number_cut = 3
        col_maxcnt = 3

        # 마이너스 '-' 표시 제대로 출력
        matplotlib.rcParams['axes.unicode_minus'] = False

        if __name__ == "__main__":
            df = pd.read_csv(common.get_local_file_path() + '회귀_나이_진료시간.csv')
        else:
            df = pd.DataFrame(request_json)

        df = df.dropna()
        if len(df.columns) == col_maxcnt:
            df_input_independent = df.loc[:, 'Predic_InputValue']
            df_input_independent = df_input_independent.dropna()
            df_input_independent = df_input_independent.reset_index(drop=True)

            df = df.iloc[:, [0, 1]]
            # df = df.dropna()
            input_independent_val = df_input_independent[0]
            input_yn = True
        else:
            input_yn = False

        df.columns = [col_indepen, col_depen]

        # 도수분포 비교를 위한 DataFrame
        origin_df = df

        # 기술통계 결과지표 DataFrame
        origin_df_copy = df.copy()

        # Exception 설명:ufunc 'true_divide' not supported for the input types, and the
        # inputs could not be safely coerced to any supported types according to the casting rule ''safe''
        origin_df_copy.replace('', np.nan, inplace=True)
        origin_df_copy = origin_df_copy.dropna()

        # origin_df_copy[col_depen] = origin_df_copy[col_depen].astype('int')  # 조건절 타입에 맞도록 변경
        origin_df_copy[col_depen] = origin_df_copy[col_depen].astype('float')  # 조건절 타입에 맞도록 변경

        for colName in df.columns:
            df[colName] = pd.to_numeric(df[colName])

        filter_param = col_depen + ' ~ ' + col_indepen

        # ols('반응변수 ~ 설명변수1+설명변수2+..', data=데이터).fit() (단순선형회귀모형 적합)
        fit = sm.OLS.from_formula(filter_param, data=df).fit()

        # 회귀분석 시각화 (분석결과 이미지 저장)
        result_Image = common.Path_Prj_Main() + word.result_image_path + 'LinearImg_' + common.Regexp_OnlyNumberbyDate() + '.png'
        font_size = word.result_font_size
        fig = plt.figure(figsize=(word.result_Img_width, word.result_Img_height))
        fig.set_facecolor('white')

        # 원 데이터 산포도
        plt.scatter(df[col_indepen], df[col_depen])
        # 회귀직선 추가
        plt.plot(df[col_indepen], fit.fittedvalues, color='red')
        plt.xlabel(col_indepen, fontsize=font_size)
        plt.ylabel(col_depen, fontsize=font_size)
        plt.savefig(result_Image)
        # plt.show()

        return_data = [round(fit.uncentered_tss, number_cut), round(fit.mse_model, number_cut),
                       round(fit.ssr, number_cut), (round(fit.rsquared, number_cut)) * 100]

        result_return_value.append(df[col_indepen])

        # 모델 성능지표
        result_func = [pfm.get_clf_eval(df[col_indepen], fit.fittedvalues),
                       pfm.eval_all(df[col_indepen], fit.fittedvalues, True, fit),
                       return_data]

        result_data = pd.DataFrame(result_return_value)

        result_data = result_data.transpose()
        result_data['RESULT'] = round(fit.fittedvalues, number_cut)

        result_data = result_data.sort_values('INDEPEN')  # line chart 이므로 소트한 결과로 리턴
        result_data.columns = ['x', 'y']

        # 중복을 제거하기 위함이지만, index 순번에 영향을 받기 때문에
        # concat을 처리할때 이빠지듯이 Nan값이 추가되는 현상으로 주석처리함
        # result_data = result_data.drop_duplicates()

        # 두 DataFrame의 결과를 병합하여 리턴 (회귀직선과 모델평가 지표 결과값)
        summary_arr = [str(fit.summary())]
        df_summary = pd.DataFrame(summary_arr)
        df_summary.rename(columns={df_summary.columns[0]: "SUMMARY"}, inplace=True)

        return_df = pd.concat([result_data, df_summary], axis=1)
        return_df = pd.concat([return_df, pd.DataFrame(result_func).transpose()], axis=1)
        return_df.columns = ['x', 'y', 'SUMMARY', '0', 'P_MODEL_RESULT', 'P_MODEL_RSQ_ERR_VAL']

        # 예측입력값이 있을 경우
        if input_yn and round(fit.predict(exog=dict(INDEPEN=[int(input_independent_val)])), number_cut)[0] > 0:
            input_val = [round(fit.predict(exog=dict(INDEPEN=[int(input_independent_val)])), number_cut)[0]]
            input_df = pd.DataFrame(input_val)
            input_df.columns = ['input_result']
            return_df = pd.concat([return_df, input_df], axis=1)

        # 정규분포 그래프
        histogram_df = pfm.make_histogram_data(origin_df)
        return_df = pd.concat([return_df, histogram_df], axis=1)

        # 기술통계 결과지표
        sts_df = pfm.descriptive_statistics(np.array(origin_df_copy[col_depen]))
        return_df = pd.concat([return_df, sts_df], axis=1)

        # 분석결과 리포트 생성
        # 결과보고서 버튼 클릭시 중복분석을 방지하기 위해서 분석실행시 보고서를 만든다.
        report_fileName = word.CreateReport_LinearRegression(return_df, result_Image)
        fileNameArr = [report_fileName]
        return_df = pd.concat([return_df, pd.DataFrame(fileNameArr, columns=["FILENAME"])], axis=1)

        return return_df.transpose()

    except Exception as err:
        common.exception_print(err)


def linear_EDA_run(request_json):
    try:
        df = pd.DataFrame(request_json)
        common.Create_EDA_Report(df)

    except Exception as err:
        common.exception_print(err)


if __name__ == "__main__":
    linear_regression_run(None)
