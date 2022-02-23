from Analytics.Association import AssociationRules
from Analytics.Clustering import DBScan, KMeans
from Analytics.TimeSeries import TimeSeries_Prophet, RollingTimeSeries, TimeSeries
from Analytics.Regression import LinearRegression, MultipleLinearRegression, RidgeRegression, LassoRegression

from typing import Any, Dict, AnyStr, List, Union

ml_Service = "/ML_Service"  # 데이터 분석관련

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]


# # 테스트 GET 랜덤숫자 리턴
# def fn_rand(app):
#     @app.get(ml_Service + "/rand")
#     def rand():
#         print(str(random.randint(0, 100)))
#         return str(random.randint(0, 100))


# 시계열 분석 : 오차정보, 밀도정보
def fn_ml_show_time_series(app):
    @app.post(ml_Service + "/ML_Show_TimeSeries")
    def ml_show_time_series(request_json_data: JSONStructure = None):
        return TimeSeries.time_series_show_data(request_json_data).to_json()


# 시계열 분석 : 예측값
def fn_ml_time_series(app):
    @app.post(ml_Service + "/ML_TimeSeries")
    def ml_time_series(request_json_data: JSONStructure = None):
        return TimeSeries.time_series_run(request_json_data).to_json()


# 시계열 분석 Propeht : 예측값
def fn_ml_time_series_prophet(app):
    @app.post(ml_Service + "/ML_TimeSeriesProphet")
    def ml_time_series_prophet(request_json_data: JSONStructure = None):
        return TimeSeries_Prophet.time_series_prophet_run(request_json_data).to_json()


# 회귀분석
def fn_ml_linear_regression(app):
    @app.post(ml_Service + "/ML_LinearRegression")
    def ml_linear_regression(request_json_data: JSONStructure = None):
        return LinearRegression.linear_regression_run(request_json_data).to_json()


# 릿지회귀분석
def fn_ml_Ridge_regression(app):
    @app.post(ml_Service + "/ML_RidgeRegression")
    def ml_Ridge_regression(request_json_data: JSONStructure = None):
        return RidgeRegression.ridgeRegression_run(request_json_data).to_json()


# 라쏘회귀분석
def fn_ml_Lasso_regression(app):
    @app.post(ml_Service + "/ML_LassoRegression")
    def ml_Lasso_regression(request_json_data: JSONStructure = None):
        return LassoRegression.lassoRegression_run(request_json_data).to_json()


# 시계열 : 이동평균
def fn_rolling_time_series_run(app):
    @app.post(ml_Service + "/ML_RollingTimeSeries")
    def rolling_time_series_run(request_json_data: JSONStructure = None):
        return RollingTimeSeries.rolling_time_series_run(request_json_data).to_json()


# 비지도 k-Means
def fn_ml_k_means(app):
    @app.post(ml_Service + "/ML_KMeans")
    def ml_k_means(request_json_data: JSONStructure = None):
        return KMeans.kmeans_run(request_json_data).to_json()


# 비지도 DB-Scan
def fn_ml_DBScan(app):
    @app.post(ml_Service + "/ML_DBScan")
    def ml_DBScan(request_json_data: JSONStructure = None):
        return DBScan.DBscan_run(request_json_data).to_json()


# 연관분석
def fn_ml_Apriori(app):
    @app.post(ml_Service + "/ML_Apriori")
    def ml_Apriori(request_json_data: JSONStructure = None):
        return AssociationRules.Apriori_Run(request_json_data).to_json()


# 상관분석 & 다중회귀분석
def fn_ml_MultipleLinear(app):
    @app.post(ml_Service + "/ML_MultipleLinearRegression")
    def ml_Correlation(request_json_data: JSONStructure = None):
        return MultipleLinearRegression.MultipleLinearRegression(request_json_data).to_json()