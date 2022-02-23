import os
import sys
import CommonLib.DBManager.MongoManager as mongo
from datetime import datetime
import requests

# PDF, JPG 이미지 생성관련 참조
import comtypes
from pathlib import Path
from contextlib import suppress
from pdf2jpg import pdf2jpg  # pip install pdf2jpg 에러 발생시 pip install --upgrade pip (pip 버전 : 21.3.1 에서 설치됨)
from comtypes.client import CreateObject

result_reportPDF_path = 'Data/Report/ML_Result_PDF/'    # 결과 PDF문서
result_reportIMG_path = 'Data/Report/ML_IMG/'           # 결과 PDF문서

# 오라클DB 접속정보
oracle_id = "dreamer"
oracle_pw = 'dsdvp'
oracle_ip = '192.168.19.247'
oracle_port = 1521
oracle_sid = 'CIT1'

# 몽고DB 접속정보
MongoDBName = "AnalysisDB"
MongoDBIP = "localhost"
MongoDBPort = "27017"


# 참조로 인해 Common에서 연결시켜준다.
def Mongo_Insert_by_DataFrame(collection, df):
    mongo.Insert_by_DataFrame(collection, df)


def Mongo_Insert_by_Dictionary(collection, dic):
    mongo.Insert_by_Dictionary(collection, dic)


def exception_print(err):
    """
    [2021.11.25 박성한]
    try 에러 발생시 오류구문 정의
    :param err:
    :return:
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print("★" * 40 + "Exception 오류발생 내용" + "★" * 40)
    print("Exception 내용:" + str(exc_type),
          "\nException 설명:" + str(err),
          "\n파일명:" + str(fname),
          "\n오류위치:" + str(exc_tb.tb_lineno))
    print("★" * 103)


def DataFrame_PrintFull(dataFrame, row=None, col=None):
    dataFrame.set_option('display.max_columns', col)  # 모든 열을 출력한다.
    dataFrame.set_option('display.max_rows', row)     # 모든 열을 출력한다.


def Create_EDA_Report(df):
    """
    탐색적 데이터 분석 EDA(Exploratory Data Analysis) 리포트 생성
    :param df:
    :return:
    """
    try:
        from pandas_profiling import ProfileReport  # pip install pandas-profiling
        # EDA Report 생성
        profile = ProfileReport(df,
                                minimal=False,
                                explorative=True,
                                title='Data Profiling',
                                plot={'histogram': {'bins': 8}},
                                pool_size=4,
                                progress_bar=False)

        # Report 결과 경로에 저장
        profile.to_file(output_file="data_profiling.html")
    except Exception as err:
        exception_print(err)


def ServerStart(channel, SendMessage):
    SendMessage += "\n=============================="
    requests.post("https://slack.com/api/chat.postMessage",
                  headers={"Authorization": "Bearer " + ReadTextFile("c:/FastAPI.txt")},
                  data={"channel": channel, "text": SendMessage})
    print(SendMessage)


def ReadTextFile(path):
    file = open(path, 'r')
    text = file.readline()
    file.close()
    return text


def get_local_file_path():
    """
    프로젝트 메인 경로 리턴
    :return:
    """
    return "C:/Users/User/Desktop/POC시연자료/"


def DataFrame_Information(df):
    print("##############################################################")
    print("================= DataFrame - head() =========================")
    print("##############################################################")
    print(df.head())
    print("##############################################################")
    print("================= DataFrame - shape  =========================")
    print("##############################################################")
    print(df.shape)
    print("##############################################################")
    print("================= DataFrame - isnull().sum() =================")
    print("##############################################################")
    print(df.isnull().sum())
    print("##############################################################")
    print("================= DataFrame - info() =========================")
    print("##############################################################")
    print(df.info())
    print("##############################################################")


def Word_Make_Sentense(paragraph, item):
    try:
        paragraph.text = paragraph.text.replace(str(item.text), "")
        run = paragraph.add_run(str(item.replaceText))
        run.font.color.rgb = item.color
        run.bold = item.bold
        run.font.size = item.fontSize
        run.font.name = '맑은 고딕'
    except Exception as err:
        exception_print(err)


def Word_Make_Image(paragraph, imagePath):
    """
    Word 문서에 작성할 이미지 정의
    :param paragraph:
    :param imagePath:
    :return:
    """
    run = paragraph.add_run()
    run.add_picture(imagePath)


def Regexp_OnlyNumberbyDate():
    """
    현재일자 공백없는 숫자만 출력
    :return:
    """
    import re
    return re.sub(r'[^0-9]', '', str(datetime.now()))


def ToDay(onlyDate=True):
    """
    현재일자 리턴
    :param onlyDate: True-현재일자 YYYY-mm-dd False-YYYY-mm-dd HH:MM
    :return:
    """
    if onlyDate:
        resultDate = str(datetime.today().strftime('%Y-%m-%d'))
    else:
        resultDate = str(datetime.today().strftime('%Y-%m-%d %H:%M'))

    return resultDate


def Path_Prj_Main(path=True):
    """
    프로젝트 메인 상대경로를 리턴
    :return:
    """
    from pathlib import Path
    if path:
        return str(str(Path(__file__).parent.parent) + '\\').replace('\\', '/')
    else:
        return str(str(Path(__file__).parent.parent) + '\\')


def front_Rexp(text, frontText, endText):
    """
    특정 위치의 문자열을 추출
    :param text: 검색 문장
    :param frontText: 추출 앞 단어
    :param endText: 추출 뒤 단어
    :return:
    """
    return text[text.find(frontText):text.find(endText)].replace(frontText, '').strip()


def end_Rexp(text, frontText):
    """
    문장의 마지막 문자열 이후의 값 추출
    :param text: 검색 문장
    :param frontText: 추출 앞 단어
    :return:
    """
    return text[text.find(frontText):].replace(frontText, '').strip()


def SendEmail(FromEmail, FromPassword, ToUser, HeaderText, bodyText, files):
    """
    메일전송
    :param FromEmail: 보낸사람 메일
    :param FromPassword: 메일 암호
    :param ToUser: 받는사람 메일
    :param HeaderText: 제목
    :param bodyText: 내용
    :param files: 첨부파일
    :return:
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.utils import formatdate
    from email.header import Header
    from email.mime.text import MIMEText

    from email.mime.base import MIMEBase
    from email.encoders import encode_base64
    from email.utils import COMMASPACE

    try:

        # 파일첨부를 위한 클래스
        msg = MIMEMultipart()

        # 보낸사람
        msg['From'] = FromEmail

        # 받는사람
        msg['To'] = COMMASPACE.join(ToUser)

        # 받은 일자
        msg['Date'] = formatdate(localtime=True)

        # 제목
        msg['Subject'] = Header(s=HeaderText, charset='utf-8')

        # 본문내용
        body = MIMEText(bodyText, _charset='utf-8')
        msg.attach(body)

        # 파일 첨부
        for f in files:
            part = MIMEBase('application', "octet-stream")
            part.set_payload(open(f, "rb").read())
            encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(f))
            msg.attach(part)

        mailServer = smtplib.SMTP_SSL('smtp.gmail.com')

        # 보낸사람 로그인 정보
        mailServer.login(FromEmail, FromPassword)
        mailServer.send_message(msg)
        mailServer.quit()

        # 메일 전송 샘플
        # FromUser = 'psh85aaa@gmail.com'
        # ToUser = list()
        # ToUser.append('seonghan.park@cwit.co.kr')
        # ToUser.append('areum.choe@cwit.co.kr')
        # files = list()
        # files.append(
        #     'D:/26.DataAnalisyPrj/ProxyServer/FastAPIServer/Data/Report/ML_Result/MLReport_Result_20220121150921434134.docx')
        # files.append(
        #     'D:/26.DataAnalisyPrj/ProxyServer/FastAPIServer/Data/Report/ML_Result/MLReport_Result_20220121153946705766.docx')
        # SendEmail(FromUser, 'ppark24', ToUser, '파일첨부 메일송신 테스트', '첨부된 파일을 확인해주세요.', files)

    except Exception as err:
        exception_print(err)


def word_to_pdf(file):
    try:
        print('word_to_pdf 변환 start')
        # 파라미터로 pdf 파일의 절대경로를 받는다.
        word = comtypes.client.CreateObject('Word.Application')
        word.Visible = False
        doc = word.Documents.Open(file)

        output_file_path = Path_Prj_Main() + result_reportPDF_path + Path(file).stem + ".pdf"

        doc.SaveAs(output_file_path, FileFormat=17)
        doc.Close()
        print('word_to_pdf 변환 end')
        return output_file_path
    except Exception as err:
        exception_print(err)


def pdf_to_jpg(file):
    try:
        import shutil
        import os

        # pdf가 여러 장으로 되어있다면 모든 장을 jpg로 바꾼다.
        pdf2jpg.convert_pdf2jpg(file, Path_Prj_Main() + result_reportPDF_path, dpi=300, pages='ALL')

        # todo 다중 페이지 일 경우 검토필요
        move_fileName = '0_' + Path(file).stem + '.pdf.jpg'
        src = Path_Prj_Main() + result_reportPDF_path + Path(file).stem + '.pdf_dir/'
        delete_dir = Path_Prj_Main() + result_reportPDF_path + Path(file).stem + '.pdf_dir'
        targetDir = Path_Prj_Main() + result_reportIMG_path

        # convert_pdf2jpg 함수로 생성된 파일위치를 변경
        shutil.move(src + move_fileName, targetDir + move_fileName)

        # convert_pdf2jpg 함수로 임의생성된 파일명칭 수정
        os.rename(targetDir + move_fileName, targetDir + Path(file).stem + '_0.jpg')

        # convert_pdf2jpg 함수로 생성된 폴더 삭제
        os.rmdir(delete_dir)
    except Exception as err:
        exception_print(err)


def word_to_jpg(file):
    with suppress(KeyError):pdf_to_jpg(word_to_pdf(file))


def UpdaetImageFlag(imageFileName, imageFlag):
    data = {"Report_File": Path(imageFileName).stem}
    update_data = {"ImageYN": imageFlag}
    mongo.Update_One('Prophet', data, update_data)


if __name__ == "__main__":
    ReadTextFile("c:/FastAPI.txt")

    # FromUser = 'psh85aaa@gmail.com'
    # ToUser = list()
    # ToUser.append('seonghan.park@cwit.co.kr')
    # ToUser.append('areum.choe@cwit.co.kr')
    # files = list()
    # files.append('D:/26.DataAnalisyPrj/ProxyServer/FastAPIServer/Data/Report/ML_Result/MLReport_Result_20220121150921434134.docx')
    # files.append(
    #     'D:/26.DataAnalisyPrj/ProxyServer/FastAPIServer/Data/Report/ML_Result/MLReport_Result_20220121153946705766.docx')
    # SendEmail(FromUser, 'ppark24', ToUser, '파일첨부 메일송신 테스트', '첨부된 파일을 확인해주세요.', files)