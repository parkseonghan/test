import CommonLib.Common as common
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import CommonLib.Performance.PerformanceManager as pfm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

common.DataFrame_PrintFull(pd)


def lassoRegression_run(request_json):
    try:
        alpha_val = 1.0

        lasso_alpha = []
        train_score = []
        test_score = []

        if __name__ == "__main__":
            df1 = pd.read_csv(common.get_local_file_path() + '상관분석_혈압나이성별온도.csv')
            df1['sex'] = df1['sex'].map({'F': 0, 'M': 1})
        else:
            df1 = pd.DataFrame(request_json)

        if 'ALPHA' in df1.columns:
            df = df1.drop(columns=['ALPHA'], axis=1)
            alpha_val = df1['ALPHA'].values[0]
        else:
            df = df1

        df = df.dropna(axis=0)
        df = df.astype('float')

        x = df.iloc[:, 1:]
        y = df.iloc[:, [0]]

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

        lasso = Lasso(alpha=alpha_val)

        lasso.fit(x_train, y_train)

        # 모델을 사용해서 만든 예측값
        y_predict = lasso.predict(x_test)

        # 실제 키 값
        y_test = y_test.reset_index(drop=True)
        y_test.columns = ['실제값']

        # 실제값 & 예측값 합치기
        result_all = pd.concat([y_test, pd.DataFrame(y_predict, columns=['예측값'])], axis=1)

        # 오차값 합치기
        result_all = pd.concat([result_all, pd.DataFrame(result_all.실제값 - result_all.예측값, columns=['오차값'])], axis=1)

        # 상관관계
        corr = pfm.Correlation(df)

        # 합치기
        result_all = pd.concat([corr, result_all], axis=1)

        # alpha 값에 따른 score 값
        # alpha 낮게 -> 규제 약하게 한다 -> 데이터에 더 많은 피팅 -> 과대적합 방향으로 진행
        # alpha 높게 -> 규제 강하게 한다 -> 모든 계수 값 0에 가깝게 -> 과소적합 방향으로 진행
        alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
        for alpha in alpha_list:
            lasso = Lasso(alpha=alpha)
            lasso.fit(x_train, y_train)
            lasso_alpha.append(alpha)
            train_score.append(lasso.score(x_train, y_train))
            test_score.append(lasso.score(x_test, y_test))

        # alpha 값 결정하기 위한 시각화화
        plt.plot(np.log10(alpha_list), train_score)
        plt.plot(np.log10(alpha_list), test_score)
        plt.xlabel('alpha')
        plt.ylabel('R^2')
        plt.show()

        # alpha값
        col_name3 = ['lasso_alpha']
        alpha = pd.DataFrame(lasso_alpha, columns=col_name3)

        # alpha 값 별로 train 과 test 의 score 를 담은 것것
        col_name1 = ['train_score_alpha']
        train_score_alpha = pd.DataFrame(train_score, columns=col_name1)

        col_name2 = ['test_score_alpha']
        test_score_alpha = pd.DataFrame(test_score, columns=col_name2)

        # 합치기
        result_all = pd.concat([result_all, pd.DataFrame(alpha)], axis=1)
        result_all = pd.concat([result_all, train_score_alpha], axis=1)
        result_all = pd.concat([result_all, test_score_alpha], axis=1)

        # 평가지표
        eval_all = pfm.eval_all(y_test, pd.DataFrame(y_predict))
        result_all = pd.concat([result_all, pd.DataFrame(eval_all, columns=['평가지표'])], axis=1)

        # 피쳐 중요도
        feature_importance = pd.DataFrame(list(zip(x_train.columns, np.array(lasso.coef_).reshape(7))),
                                          columns=['features', 'importances']).sort_values('importances')

        # 합치기
        result_all = pd.concat([result_all, feature_importance], axis=1)


        return result_all

    except Exception as err:
        common.exception_print(err)


if __name__ == "__main__":
    lassoRegression_run(None)
