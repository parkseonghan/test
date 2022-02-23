# evaluation metrics 참고 url : https://rfriend.tistory.com/671
from sklearn.metrics import mean_squared_error as metrics_mse
from sklearn.metrics import mean_absolute_error as metrics_mae
from sklearn.metrics import mean_squared_log_error as metrics_msle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, r2_score
import CommonLib.Common as common

# 기술통계 결과지표
from scipy.stats import skew, kurtosis

# 히스토그램 관련 import
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 정규분포 관련
from scipy.stats import norm

number_cut = 3


def descriptive_statistics(array_val):
    """
    [2021.11.30 박성한]
    기술통계 지표결과
    ex) df = descriptive_statistics(np.array(df['ORDERTIME']))

    :param array_val: array value
    :return: DataFrame 결과리턴
    """
    try:
        # 리스트로 변환 (왜도, 첨도)
        x_list = array_val.tolist()

        statistics_result_arr = [["평균", "중앙값", "누적 합", "최대값", "최소값", "범위", "분산", "표준 편차", "1분위수(25%)",
                                  "2분위수(50%)", "3분위수(75%)", "첨도", "왜도"],
                                 [str(round(array_val.mean(), number_cut)),
                                  str(round(np.median(array_val), number_cut)),
                                  str(round(array_val.cumsum()[-1], number_cut)),
                                  str(array_val.max()),
                                  str(array_val.min()),
                                  str(round(array_val.max() - array_val.min(), number_cut)),
                                  str(round(array_val.var(), number_cut)),
                                  str(round(array_val.std(), number_cut)),
                                  str(np.percentile(array_val, 25)),
                                  str(np.percentile(array_val, 50)),
                                  str(np.percentile(array_val, 75)),
                                  str(round(kurtosis(x_list), number_cut)),
                                  str(round(skew(x_list), number_cut))]]

        df_return = pd.DataFrame(statistics_result_arr).transpose()
        df_return.columns = ["type", "sts_result"]

        return df_return

    except Exception as err:
        common.exception_print(err)


def create_normal_distribution(bins_cnt, linewidth, array_data):
    """
    [2021.11.29 박성한]
    확률밀도함수(probability density function) 데이터 생성
    :param linewidth: line chart witdh
    :param bins_cnt:그룹카우트
    :param array_data:데이터
    :return: 정규분포 결과 DataFrame
    """
    try:
        # 평균, 표준편차
        mu, std = norm.fit(array_data)

        # Starting a Matplotlib GUI outside of the main thread will likely fail
        if __name__ == "__main__":
            plt.hist(array_data, bins=bins_cnt, density=True, alpha=0.6, color='g')

        # np.linspace(() min~max 까지 bins_cnt개의 구간 설정
        # xmin, xmax를 넘겨받은 array의 값으로 수정 -> 이전 데이터값을 물고나와서 문제 발생됨
        # x_val = np.linspace(xmin, xmax, bins_cnt)
        x_val = np.linspace(np.array(array_data).min(), np.array(array_data).max(), bins_cnt)

        # pdf 확률밀도함수(probability density function)
        predic = norm.pdf(x_val, mu, std)

        title = "Fit results : mu=%.2f, std=%.2f" % (mu, std)
        title_result = [title]

        df_result = pd.concat([pd.DataFrame(x_val), pd.DataFrame(predic), pd.DataFrame(title_result)], axis=1)
        df_result.columns = ['x_nmd', 'y_nmd', 'title']

        if __name__ == "__main__":
            plt.plot(x_val, predic, 'k', linewidth=linewidth)
            plt.title(title)
            plt.show()

        return df_result

    except Exception as err:
        common.exception_print(err)


def make_histogram_data(df, binscount=17):
    """
    [2021.11.25 박성한]
    도수분포를 그릴 데이터를 형성한다

    :param binscount: 클래스 카운트
    :param df: 히스토그램 요청 Data원본
    :return:도수분포 결과 DataFrame
    """
    try:
        if __name__ == "__main__":
            df = pd.read_csv(common.get_local_file_path() + '회귀_나이_진료시간.csv')

        # 히스토그램 그룹화 카운트 (font-end 차트와 동일하게 처리)
        bins_cnt = binscount

        # dependent 기준으로 도수분포 데이터를 만들었습니당
        # independent  독립변수 X
        # dependent    종속변수 Y
        col_name1 = 'independent'
        col_name2 = 'dependent'
        df.columns = [col_name1, col_name2]
        df = df.dropna()

        # A value is trying to be set on a copy of a slice from a DataFrame.
        # https://emilkwak.github.io/pandas-dataframe-settingwithcopywarning
        # 복사본의 DataFrame을 추가 후 복사된 데이터 프레임을 변경하자 -> 깊은 복사하라는 얘기
        # df[col_name2] = df[col_name2].astype('int')
        df_tmp = df.copy(deep=True)
        df_tmp[col_name2] = df_tmp[col_name2].astype('int')  # 조건절 타입에 맞도록 변경

        min_val = np.min(df_tmp[col_name2])
        max_val = np.max(df_tmp[col_name2])

        # 계급을 나누기 위한 기준값 설정 (최대-최소)/그룹수
        # np.linspace(최소, 최대, 계급수) 해당 함수 사용하면 한줄로 끝남... 뒤늦게 좋을걸 발견했듬.. 참고하자.
        standard_num = (max_val - min_val) / bins_cnt

        # 클래스 그룹배열
        calss_result = []
        class_val = min_val + standard_num
        calss_result.append([int(min_val), int(class_val)])

        # 기준값으로 그룹 배열을 정의
        for index in range(2, bins_cnt + 1):
            before_val = class_val
            class_val = class_val + standard_num
            calss_result.append([int(before_val)+1, int(class_val)])

        # 클래스별 분포 카운트 리스트 선언
        result_histogram = []

        # 클래스별 카운트 설정
        for item in calss_result:
            count = 0
            for index, var in enumerate(df_tmp[col_name2]):
                if item[0] <= var <= item[1]:
                    count += 1
                if index == len(df_tmp) - 1:
                    result_histogram.append([item[0], item[1], count])

        result_df = pd.DataFrame(result_histogram)
        result_df.columns = ['start_range', 'end_range', 'count']

        # 정규분포(확률밀도함수) 데이터 설정
        df_normal = create_normal_distribution(bins_cnt, 2, df_tmp[col_name2])
        result_df = pd.concat([result_df, df_normal], axis=1)

        if __name__ == "__main__":
            sns.displot(df_tmp[col_name2], bins=bins_cnt, color="orange")
            plt.show()

        return result_df

    except Exception as err:
        common.exception_print(err)


def sse(y_test, y_pred):
    try:
        y_test, y_pred = np.array(y_test), np.array(y_pred)
        return np.sum((y_test - y_pred) ** 2)
    except Exception as err:
        common.exception_print(err)


def me(y_test, y_pred):
    try:
        y_test, y_pred = np.array(y_test), np.array(y_pred)
        return np.mean(y_test - y_pred)
    except Exception as err:
        common.exception_print(err)


def rmse(y_test, y_pred):
    try:
        y_test, y_pred = np.array(y_test), np.array(y_pred)
        return np.sqrt(np.mean((y_test - y_pred) ** 2))
    except Exception as err:
        common.exception_print(err)


def mpe(y_test, y_pred):
    try:
        y_test, y_pred = np.array(y_test), np.array(y_pred)
        return np.mean((y_test - y_pred) / y_test) * 100
    except Exception as err:
        common.exception_print(err)


def mape(y_test, y_pred):
    try:
        y_test, y_pred = np.array(y_test), np.array(y_pred)
        return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    except Exception as err:
        common.exception_print(err)


def get_clf_eval(y_test, y_pred=None, nan_check=False):
    """
    [2021.11.25 박성한]
    분류모델 성능 평가지표

    :param y_test: 훈련데이터
    :param y_pred: 예측데이터
    :param nan_check: 결측치 제거 유무
    :return:
    """
    try:
        # 소수점 버림
        y_test = np.floor(y_test)
        y_pred = np.floor(y_pred)

        # 결측치 제거
        if nan_check:
            y_test, y_pred = check_nan_data(y_test, y_pred)

        eval_test = np.array(y_test)
        eval_predic = np.array(y_pred)

        eval_test = eval_test[eval_test != 0]  # 0이 들어간 데이터는 제외

        # 배열의 사이즈를 동일하게 처리하기 위해 마지막 인덱스 조정
        if eval_test.size > eval_predic.size:
            y_test = np.delete(eval_test, eval_test.size - 1)
        elif eval_predic.size > eval_test.size:
            y_pred = np.delete(eval_predic, eval_predic.size - 1)

        # average=None:구한 각 열의 Precision들을 산술 평균한 값이 macro가된다.
        # macro:각 열에 대한 precision 값을 모두 더한 다음 열의 갯수로 나눈 것
        # micro:전체 평균으로 모든 열에서 맞은 것 즉, 대각선 성분의 총 합을 총 갯수로 나눈 것
        accuracy = round(accuracy_score(y_test, y_pred), number_cut)  # 정확도
        precision = round(precision_score(y_test, y_pred, average='micro'), number_cut)  # 정밀도
        recall = round(recall_score(y_test, y_pred, average='micro'), number_cut)  # 재현율
        f1score = round(f1_score(y_test, y_pred, average='micro'), number_cut)  # F1 Score

        return [accuracy, precision, recall, f1score]

    except Exception as err:
        common.exception_print(err)


def check_nan_data(y_test, y_pred):
    if y_test.isnull().sum().sum() > 0:
        y_test = y_test.dropna()

    if y_pred.isnull().sum().sum() > 0:
        y_pred = y_pred.dropna()

    return y_test, y_pred


def eval_all(y_test, y_pred, nan_check=False, fit=None):
    """
    [2021.11.25 박성한]
    모델 성능지표에 대한 결과 반환
    모델 성능평가 지표 중에서 실제값과 예측값의 차이인 잔차를 기반으로 계산
    낮을수록 상대적으로 더 좋은 모델지표
    MSE, RMSE, ME, MAE, MPE, MAPE, AIC, SBC, APC

    :param fit:
    :param y_test: 원본 데이터
    :param y_pred: 예측결과 데이터
    :param nan_check: 결측치 제거 유무
    :return:
    """
    try:
        # 소수점 버림
        y_test = np.floor(y_test)
        y_pred = np.floor(y_pred)

        # 결측치 제거
        if nan_check:
            y_test, y_pred = check_nan_data(y_test, y_pred)

        # Adjusted-R2는 값이 높을 수록 상대적으로 더 좋은 모델평가
        # Ajd.R2가 음수가 나온 값은 예측 모델 불량으로 인해
        # 예측값과 실제값이 차이가 큼에 따라 SSE가 SST보다 크게 되었기 때문.
        if fit is not None:
            rst_radj = round(fit.rsquared, number_cut) * 100
            rst_msle = round(metrics_msle(y_test, abs(fit.predict(y_test))), number_cut)
        else:
            rst_radj = (round(r2_score(y_test, y_pred), number_cut)) * 100
            rst_msle = round(metrics_msle(y_test, y_pred), number_cut)

        rst_sse = round(sse(y_test, y_pred), number_cut)
        rst_mse = round(metrics_mse(y_test, y_pred), number_cut)
        rst_rmse = round(np.sqrt(rst_mse), number_cut)
        # 오류로 인한 수정 abs: Mean Squared Logarithmic Error cannot be used when targets contain negative values.

        rst_me = round(me(y_test, y_pred), number_cut)
        # rst_mae = round((metrics_mae(y_test, y_pred) / y_test.mean()) * 100, number_cut)
        rst_mae = round(metrics_mae(y_test, y_pred), number_cut)

        # print(rst_mae)
        # print(np.abs(np.subtract(np.array(y_test), np.array(y_pred))).mean())
        # print(metrics_mae(y_test, y_pred))
        # print(metrics_mse(y_test, y_pred))
        # print(100*'=')

        rst_mpe = round(mpe(y_test, y_pred), number_cut)
        rst_mape = round(mape(y_test, y_pred), number_cut)

        # print("모델 성능지표")
        # print(100*'=')
        # print("rst_radj:" + str(rst_radj) + " \nrst_mae:" + str(rst_mae) + " \nrst_mse:" + str(rst_mse) +
        #       "\nrst_rmse:" + str(rst_rmse) + " \nrst_msle:" + str(rst_msle) + " \nrst_mpe:" + str(rst_mpe) +
        #       "\nrst_mape:" + str(rst_mape) + " \nrst_me:" + str(rst_me) + " \nrst_sse:" + str(rst_sse))
        # print(100*'=')

        return [rst_radj, rst_mae, rst_mse, rst_rmse, rst_msle, rst_mpe, rst_mape, rst_me, rst_sse]

    except Exception as err:
        common.exception_print(err)


def Correlation(df):
    """
    상관관계 메트릭스 DataFrame 리턴
    :param df:
    :return:
    """
    df_corr = df.corr()
    df_corr = df_corr.apply(lambda x: round(x, 2))
    corr_df2 = df_corr.reset_index()

    return corr_df2


if __name__ == "__main__":
    make_histogram_data(None, 10)
