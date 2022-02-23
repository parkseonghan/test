import CommonLib.Common as common
import CommonLib.DBManager.MongoManager as mongo
import CommonLib.Report.WordManager as word
import pandas as pd
import DAO.TimeSeriesDAO as time

from typing import Any, Dict, AnyStr, List, Union

DB_Service = "/DB_Service/"
ML_Service = "/ML_Service/"

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]


def GetReportData(mongoDB, userID, collection):
    """
    mongoDB에 저장된 데이터 리턴
    :param mongoDB: DB객체
    :param userID: 사용자 ID
    :param collection: 컬렉션명칭
    :return:
    """
    try:
        from pathlib import Path
        # list_Prophet = mongoDB.Select_find(collection, {word.table_col_userid: userID})
        list_Prophet = mongoDB.Select_find(collection, {word.table_col_userid: userID, "ImageYN":"Y"})
        if len(list_Prophet) > 0:
            df = pd.DataFrame(list_Prophet)
            df = df[[word.table_col_title, word.table_col_file, word.table_col_writedate, word.table_col_userid]]
            return df
        else:
            return None
    except Exception as err:
        common.exception_print(err)


def fn_Get_ReportList(app):
    @app.post(DB_Service + "GetReportList")
    def GetReportList():
        try:
            # todo 요청한 사용자ID의 정보로 변경해야함.
            userID = "seonghan.park@cwit.co.kr"
            mongoDB = mongo.MongoDB_Con(common.MongoDBName, common.MongoDBIP, common.MongoDBPort)

            df_Prophet = GetReportData(mongoDB, userID, word.collection_Prophet)
            df_Linear = GetReportData(mongoDB, userID, word.collection_Linear)

            if df_Prophet is None and df_Prophet is None:
                return None

            reult_df = pd.concat([df_Prophet, df_Linear])
            reult_df.sort_values(word.table_col_writedate, ascending=False, inplace=True)
            reult_df.reset_index(drop=True, inplace=True)
            reult_df = reult_df.transpose()
            return reult_df.to_json()
        except Exception as err:
            common.exception_print(err)


def fn_Oracle_RollingTimeSeries(app):
    @app.post(ML_Service + "ML_RollingTimeSeries_Oracle")
    def GetRollingTimeSeriesOracle(request_json_data: Dict):
        try:
            query_result = time.GetTimeSeriesData(request_json_data)
            df = pd.DataFrame(query_result)
            import Analytics.TimeSeries.RollingTimeSeries as rolling
            return rolling.rolling_time_series_run(None, df).to_json()
        except Exception as err:
            common.exception_print(err)


def fn_Oracle_TimeSeries_Prophet(app):
    @app.post(DB_Service + "ML_TimeSeriesProphet_Oracle")
    def GetTimeSeriesProphetOracle(request_json_data: Dict):
        try:
            query_result = time.GetTimeSeriesData(request_json_data)
            df = pd.DataFrame(query_result)
            import Analytics.TimeSeries.TimeSeries_Prophet as prophet
            return prophet.time_series_prophet_run(None, df).to_json()
        except Exception as err:
            common.exception_print(err)


if __name__ == "__main__":
    userID = "seonghan.park@cwit.co.kr"
    mongoDB = mongo.MongoDB_Con(common.MongoDBName, common.MongoDBIP, common.MongoDBPort)

    df_Prophet = GetReportData(mongoDB, userID, word.collection_Prophet)
    df_Linear = GetReportData(mongoDB, userID, word.collection_Linear)

    reult_df = pd.concat([df_Prophet, df_Linear])
    reult_df.reset_index(drop=True, inplace=True)
    reult_df = reult_df.transpose()
    reult_df.to_json()
