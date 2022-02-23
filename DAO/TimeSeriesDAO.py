import CommonLib.DBManager.OracleManager as OracleCon
import CommonLib.Common as common
import pandas as pd


def GetParamSetting(request_json_data):
    tableName = request_json_data["TableName"]
    startColName = request_json_data["StartDateColumn"]
    endColName = request_json_data["EndDateColumn"]
    strDate = request_json_data["StartDate"]
    endDate = request_json_data["EndDate"]

    return tableName, startColName, endColName, strDate, endDate


def GetTimeSeriesData(request_json_data):
    try:
        # tableName = request_json_data["TableName"]
        # startColName = request_json_data["StartDateColumn"]
        # endColName = request_json_data["EndDateColumn"]
        # strDate = request_json_data["StartDate"]
        # endDate = request_json_data["EndDate"]
        tableName, startColName, endColName, strDate, endDate = GetParamSetting(request_json_data)
        OracleMng = OracleCon.cls_Oracle(common.oracle_id, common.oracle_pw, common.oracle_ip, common.oracle_port,
                                         common.oracle_sid)
        with OracleMng.oracle_connection() as oraConnection:
            query = """ SELECT %s, COUNT(*) AS VAL
                      FROM %s
                     WHERE 1 = 1
                       AND %s >= '%s'
                       AND %s <= '%s'
                  GROUP BY %s
                  ORDER BY %s ASC """
            query = query % (
                startColName, tableName, startColName, strDate, endColName, endDate, startColName, startColName)
            print(query)
            return pd.read_sql(query, oraConnection).copy()
    except Exception as err:
        common.exception_print(err)
