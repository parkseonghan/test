import CommonLib.Common as common
import numpy as np
import pandas as pd
import CommonLib.Report.WordManager as word
import CommonLib.Performance.PerformanceManager as pfm
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier


def LogisticRegression_Run(request_json):
    try:
        if __name__ == "__main__":
            df = pd.read_csv(common.get_local_file_path() + '로지스틱.csv')
        else:
            df = pd.DataFrame(request_json)

        df = df.dropna(axis=0)
        df = df.astype('float')

        x = df.iloc[:, 1:]   # 독립변수들
        y = df.iloc[:, [0]]  # 종속변수
        y = y.astype('int')

        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=1)

        model = sm.Logit(train_y, train_x)
        results = model.fit()

        # summary - coef 값 : 계수값으로 0에 가까울수록 종속변수에 미치는 영향이 적고 멀수록 영향력이 있음. (ex:성별이 종속변수 일 경우 키가 영향력이 높음)
        summary_arr = [str(results.summary())]
        df_summary = pd.DataFrame(summary_arr)
        df_summary.rename(columns={df_summary.columns[0]: "SUMMARY"}, inplace=True)

        # 오즈비(Odds Ratio) 출력 - height을 예로 들어보면 키가 커질 수록 남자가 될 확률이 높아진다는 뜻
        params = pd.DataFrame(np.exp(results.params)).reset_index()
        params.columns = ['Params', 'Odds Ratio']

        # 합치기
        result_all = pd.concat([df_summary, params], axis=1)

        # 모델 적용 전 정규화
        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        test_x_scaled = scaler.transform(test_x)

        # 모델 적용
        lr = LogisticRegression()
        lr.fit(train_x_scaled, train_y)

        # train 정확도
        train_score = lr.score(train_x_scaled, train_y)

        # test 정확도
        test_score = lr.score(test_x_scaled, test_y)

        # 예측정확도
        y_pred = lr.predict(test_x_scaled)
        acc = metrics.accuracy_score(test_y, y_pred)
        result_all = pd.concat([result_all, pd.DataFrame(list(np.array(acc).reshape(1)), columns=['Acc'])], axis=1)

        # 모델 피처 중요도
        # 어느 데이터가 확률값 계산에 중요한 작용을 했는지 나타내는 지표
        # 수치가 높은것은 중요도가 높으며 영향력이 크다고 해석 할 수 있음.)
        feature_importance = pd.DataFrame(list(zip(train_x.columns, np.array(lr.coef_).reshape(7))),
                                          columns=['Features', 'Importances']).sort_values('Importances')

        result_all = pd.concat([result_all, feature_importance], axis=1)

        # 평가지표
        eval_all = pfm.eval_all(test_y, pd.DataFrame(y_pred))
        result_all = pd.concat([result_all, pd.DataFrame(eval_all, columns=['평가지표'])], axis=1)

        # 결정 트리
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(x, y)
        result_Image = common.Path_Prj_Main() + word.result_image_path + 'LogisticTreeImg_' + common.Regexp_OnlyNumberbyDate() + '.png'
        plt.figure(figsize=(16, 8))
        plot_tree(dt, filled=True, feature_names=list(x.columns.values))
        plt.savefig(result_Image, bbox_inches='tight', pad_inches=1)
        plt.show()

        # sklearn에서 ROC 패키지 활용
        fpr, tpr, thresholds = metrics.roc_curve(test_y, y_pred, pos_label=1)

        # ROC curve
        result_Image_Roc = common.Path_Prj_Main() + word.result_image_path + 'LogisticRocImg_' + common.Regexp_OnlyNumberbyDate() + '.png'
        plt.plot(fpr, tpr)
        plt.savefig(result_Image_Roc, bbox_inches='tight', pad_inches=1)
        plt.show()

        # AUC
        auc = np.trapz(tpr, fpr)
        # AUC는 1에 가까울수록 모델의 성능이 좋은 것이며, ROC curve는 (0,1)로 그래프가 가까이 갈 수록 정확도가 좋은 것

        result_all = pd.concat([result_all, pd.DataFrame(list(np.array(auc).reshape(1)), columns=['AUC'])], axis=1)
        result_all.to_excel("logistic_test123.xlsx")
        print(result_all)

        return result_all

    except Exception as err:
        common.exception_print(err)


if __name__ == "__main__":
    LogisticRegression_Run(None)

# import numpy as np
# import statsmodels.api as sm
# import pandas as pd
# import CommonLib.Common as common
# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
#
# def LogisticRegression_Run(request_json):
#
#     try:
#         if __name__ == "__main__":
#             df = pd.read_csv(common.get_local_file_path() + '상관분석_혈압나이성별온도.csv')
#             df['sex'] = df['sex'].map({'F': 0, 'M': 1})
#         else:
#             df = pd.DataFrame(request_json)
#
#         df = df.dropna()
#         df['height'] = df['height'].astype(int)
#         df['weight'] = df['weight'].astype(int)
#         df['temperature'] = df['temperature'].astype(int)
#         df['pulse'] = df['pulse'].astype(int)
#         df[df["sex"] == 0]
#
#         feature_columns = df.columns.difference(["sex"])
#         X = df[feature_columns]
#         y = df["sex"]
#
#         train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, train_size=0.7, test_size=0.3, random_state=1)
#         print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
#
#         model = sm.Logit(train_y, train_x)
#         results = model.fit()
#         results.summary()
#
#         np.exp(results.params)  # 오즈 비(Odds Ratio) 출력
#
#         # Odds Ratio
#         # Odds Ratio란 Odds의 비율이다. Odds란 성공/실패와 같이 상호 배타적이며 전체를 이루고 있는 것들의 비율을 의미
#         lr = LogisticRegression()  # 로지스틱 회귀 모델의 인스턴스를 생성
#         lr.fit(train_x, train_y)   # 로지스틱 회귀 모델의 가중치를 학습
#
#         Y_pred = lr.predict(test_x)
#
#         accuracy_score(test_y, Y_pred)
#
#         feature_columns = df.columns.difference(["sex"])
#         data = df[feature_columns].to_numpy()
#         target = df["sex"].to_numpy()
#
#         train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
#
#         ss = StandardScaler()
#         ss.fit(train_input)
#
#         train_scaled = train_input
#         test_scaled = test_input
#
#         lr = LogisticRegression()
#         lr.fit(train_scaled, train_target)
#
#         dt = DecisionTreeClassifier(max_depth=3, random_state=42)
#         dt.fit(train_input, train_target)
#
#         result_Image = common.Path_Prj_Main() + word.result_image_path + 'TreeImg_' + common.Regexp_OnlyNumberbyDate() + '.png'
#
#         plt.figure(figsize=(10, 10))
#         plot_tree(dt, filled=True,
#                   feature_names=['age', 'diastolic', 'height', 'pulse', 'systolic', 'temperature', 'weight'])
#         plt.savefig(result_Image, bbox_inches='tight', pad_inches=1)
#         plt.show()
#
#         return result_Image
#     except Exception as err:
#         common.exception_print(err)
#
#
# if __name__ == "__main__":
#     LogisticRegression_Run(None)