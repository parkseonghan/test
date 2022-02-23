import CommonLib.Common as common

import cx_Oracle
import os


class cls_Oracle:
    """
    [2021.12.07 박성한]
    오라클 연결 및 실행결과 리턴
    """
    id = None
    pw = None
    ip = None
    port = None
    sid = None

    def __init__(self, param_id, param_pw, param_ip, param_port, param_sid):
        try:
            self.id = param_id
            self.pw = param_pw
            self.ip = param_ip
            self.port = param_port
            self.sid = param_sid

            os.putenv('NLS_LANG', 'KOREAN_KOREA.KO16KSC5601')
            oracle_Location = r"D:\oracle_Client\instantclient_19_9"
            os.environ["PATH"] = oracle_Location + ";" + os.environ["PATH"]

            print("id:" + str(self.id) + " pw:" + str(self.pw) + " ip:" + str(self.ip) + " port:" + str(self.port) + " sid:" + str(self.sid))

        except Exception as err:
            common.exception_print(err)

    def oracle_connection(self):
        try:
            oraDsn = cx_Oracle.makedsn(host=self.ip, port=self.port, sid=self.sid)
            return cx_Oracle.connect(user=self.id, password=self.pw, dsn=oraDsn)
        except Exception as err:
            common.exception_print(err)

    def executeQuery(self, param_Query):
        try:
            ora_Dsn = cx_Oracle.makedsn(host=self.ip, port=self.port, sid=self.sid)
            connection = cx_Oracle.connect(user=self.id, password=self.pw, dsn=ora_Dsn)

            cursor = connection.cursor()
            cursor.execute(param_Query)
            result_val = str(cursor.fetchon()[0])

            cursor.close()
            connection.close()

            return result_val

        except Exception as err:
            common.exception_print(err)