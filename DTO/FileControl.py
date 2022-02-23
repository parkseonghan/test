import os
import pandas as pd
import CommonLib.Common as common
import CommonLib.Report.WordManager as word

from fastapi.responses import FileResponse
from fastapi import Request
from typing import Any, Dict, AnyStr, List, Union

ml_Service = "/FL_Service"

EDA_ProfilingFile_Path = 'Data/EDA/Profiling/'
EDA_Sweetviz_Path = 'Data/EDA/Sweetviz/'
ResultImageFileDirPath = 'Data\\Image\\ReportResult'

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]


def fn_ml_CreateHtml_EDA(app):
    @app.post(ml_Service + "/ML_CreateEDA_Report")
    def create_EDAreport(request_json_data: JSONStructure = None):
        """
        탐색적 데이터분석(EDA Exploratory Data Analysis)
        :param request_json_data:
        :return:
        """
        from pandas_profiling import ProfileReport
        try:
            htmlFileName = 'data_profiling.html'
            df = pd.DataFrame(request_json_data)
            profile = ProfileReport(df, minimal=False,
                                    explorative=True,
                                    title='Data Profiling',
                                    plot={'histogram': {'bins': 7}},
                                    pool_size=4,
                                    progress_bar=True)

            profile.to_file(output_file=os.path.join(common.Path_Prj_Main() + EDA_ProfilingFile_Path + htmlFileName))

            return {'CREATE': 'OK', 'FileName': htmlFileName}
        except Exception as err:
            common.exception_print(err)
            return {'CREATE': 'FAILE'}


def fn_ml_ServerStart(app):
    @app.on_event("startup")
    def DeleteFileList():
        try:
            print("FastAPI Server Start!")
            common.ServerStart("#analisys-start", "FastAPI Server Start! " + common.ToDay(onlyDate=False))
        except Exception as err:
            common.exception_print(err)


def fn_ml_ServerShutdown(app):
    @app.on_event("shutdown")
    def DeleteFileList():
        try:
            print("FastAPI Server shutdown!")
            common.ServerStart("#analisys-shutdown", "FastAPI Server shutdown! " + common.ToDay(onlyDate=False))
        except Exception as err:
            common.exception_print(err)


def fn_ml_SelectHtml_EDA(app):
    @app.post(ml_Service + '/ML_GetEDA_Report')
    def Select_EDAreport(request_json_data: Dict):
        try:
            fileName = request_json_data['htmlFileName']
            return FileResponse(common.Path_Prj_Main() + EDA_ProfilingFile_Path + fileName)
        except Exception as err:
            common.exception_print(err)


def fn_ml_CreateHtml_Sweetviz(app):
    @app.post(ml_Service + "/ML_CreateSweetviz_Report")
    def create_Sweetviz(request_json_data: JSONStructure = None):
        import sweetviz as sv
        try:
            htmlFileName = 'sweetviz_Advertising.html'
            df = pd.DataFrame(request_json_data)
            adver_report = sv.analyze(df)
            adver_report.show_html(common.Path_Prj_Main() + EDA_Sweetviz_Path + htmlFileName, open_browser=False)
            return {'CREATE': 'OK'}
        except Exception as err:
            common.exception_print(err)
            return {'CREATE': 'FAILE'}


def fn_ml_SelectHtml_Sweetviz(app):
    @app.post(ml_Service + '/ML_GetSweetviz_Report')
    def Select_Sweetviz(request_json_data: Dict):
        try:
            fileName = request_json_data['htmlFileName']
            return FileResponse(common.Path_Prj_Main() + EDA_Sweetviz_Path + fileName)
        except Exception as err:
            common.exception_print(err)


def fn_ml_AnalysisResultReport(app):
    @app.get(ml_Service + '/ML_AnalysisResultReport')
    def Select_ResultReport(request: Request):
        try:
            fileName = request.query_params['ReportFileName']
            common.word_to_jpg(fileName)  # 분석결과 실행시 pdf, jpg 이미지 생성
            common.UpdaetImageFlag(fileName, "Y")
            return FileResponse(fileName)
        except Exception as err:
            common.exception_print(err)


def fn_ml_SelectImage(app):
    @app.get(ml_Service + '/ML_ImageDown')
    def Select_ImageDown(request: Request):
        """
        요청된 파일명으로 저장된 보고서의 이미지 리턴
        :param request:
        :return:
        """
        try:
            fileName = common.Path_Prj_Main() + word.result_reportImg_path + request.query_params['ImageFileName'] + "_0.jpg"
            if os.path.isfile(fileName):
                return FileResponse(fileName)
            else:
                return None
        except Exception as err:
            common.exception_print(err)