# Server
import uvicorn
from fastapi import FastAPI  # , APIRouter
from starlette.middleware.cors import CORSMiddleware

import DTO.RequestData as Dto_RequestData
import DTO.Analisis as Dto_Analisis
import DTO.FileControl as File_control

# json 파싱
from typing import Any, Dict, AnyStr, List, Union

app = FastAPI()

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

# 데이터 분석
Dto_Analisis.fn_ml_show_time_series(app)
Dto_Analisis.fn_ml_time_series(app)
Dto_Analisis.fn_ml_time_series_prophet(app)
Dto_Analisis.fn_rolling_time_series_run(app)

Dto_Analisis.fn_ml_k_means(app)
Dto_Analisis.fn_ml_DBScan(app)
Dto_Analisis.fn_ml_Apriori(app)

Dto_Analisis.fn_ml_linear_regression(app)
Dto_Analisis.fn_ml_Ridge_regression(app)
Dto_Analisis.fn_ml_Lasso_regression(app)
Dto_Analisis.fn_ml_MultipleLinear(app)

File_control.fn_ml_CreateHtml_EDA(app)
File_control.fn_ml_SelectHtml_EDA(app)
File_control.fn_ml_CreateHtml_Sweetviz(app)
File_control.fn_ml_SelectHtml_Sweetviz(app)
File_control.fn_ml_AnalysisResultReport(app)
File_control.fn_ml_SelectImage(app)
File_control.fn_ml_ServerStart(app)

# Get 데이터
Dto_RequestData.fn_Oracle_TimeSeries_Prophet(app)
Dto_RequestData.fn_Oracle_RollingTimeSeries(app)
Dto_RequestData.fn_Get_ReportList(app)

# @app.on_event("startup")
# async def startup_event():
#     app.state.executor = ProcessPoolExecutor()
# @app.on_event("shutdown")
# async def on_shutdown():
#     app.state.executor.shutdown()

app = CORSMiddleware(
    app=app,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    # uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)  # 해당 IP변경되야 다른 자리에서도 서버접속 가능함.
    uvicorn.run("main:app", host="192.168.19.131", port=8000, reload=True)
