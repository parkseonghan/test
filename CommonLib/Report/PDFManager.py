

# def rotate_pages(pdf_path):
#     try:
#         pdf_writer = PdfFileWriter()
#         pdf_reader = PdfFileReader(pdf_path)
#         # Rotate page 90 degrees to the right
#         page_1 = pdf_reader.getPage(0).rotateClockwise(90)
#         pdf_writer.addPage(page_1)
#         # Rotate page 90 degrees to the left
#         page_2 = pdf_reader.getPage(1).rotateCounterClockwise(90)
#         pdf_writer.addPage(page_2)
#         # Add a page in normal orientation
#         pdf_writer.addPage(pdf_reader.getPage(0))
#
#         with open('rotate_pages.pdf', 'wb') as fh:
#             pdf_writer.write(fh)
#     except Exception as err:
#         common.exception_print(err)
#
#
# if __name__ == '__main__':
#     path = 'test2.pdf'
#     rotate_pages(path)

# PDF 이미지파일 교체 실행확인 필요.
# import sys
# import os
#
# from PIL import Image
#
#
# # Include the \n to ensure extact match and avoid partials from 111, 211...
# OBJECT_ID = "\n11 0 obj"
#
# def replace_image(filepath, new_image):
#     f = open(filepath, "r")
#     contents = f.read()
#     f.close()
#
#     image = Image.open(new_image)
#     width, height = image.size
#     length = os.path.getsize(new_image)
#
#     start = contents.find(OBJECT_ID)
#     stream = contents.find("stream", start)
#     image_beginning = stream + 7
#
#     # Process the metadata and update with new image's details
#     meta = contents[start: image_beginning]
#     meta = meta.split("\n")
#     new_meta = []
#     for item in meta:
#         if "/Width" in item:
#             new_meta.append("/Width {0}".format(width))
#         elif "/Height" in item:
#             new_meta.append("/Height {0}".format(height))
#         elif "/Length" in item:
#             new_meta.append("/Length {0}".format(length))
#         else:
#             new_meta.append(item)
#     new_meta = "\n".join(new_meta)
#     # Find the end location
#     image_end = contents.find("endstream", stream) - 1
#
#     # read the image
#     f = open(new_image, "r")
#     new_image_data = f.read()
#     f.close()
#
#     # recreate the PDF file with the new_sign
#     with open(filepath, "wb") as f:
#         f.write(contents[:start])
#         f.write("\n")
#         f.write(new_meta)
#         f.write(new_image_data)
#         f.write(contents[image_end:])
#
#
# if __name__ == "__main__":
#     if len(sys.argv) == 3:
#         replace_image(sys.argv[1], sys.argv[2])
#     else:
#         print("Usage: python process.py <pdfile> <new_image>")

# PDF 파일 merge
# from reportlab.pdfgen import canvas
# from PyPDF2 import PdfFileWriter, PdfFileReader
#
# try:
#     # 워터마크 PDF 파일 생성 (해더와 바텀에 템플릿 용도의 PDF를 생성)
#     c = canvas.Canvas('watermark.pdf')
#     c.drawImage('newsCrawling.png', 30, 520)   # top
#     c.drawImage('newsCrawling.png', 30, 15)    # bottom
#
#     for pos in range(1, 1000, 10):
#         c.drawString(15, pos, str(pos) + ' 중외정보기술 www.cwit.co.kr')  # 한글 깨짐. 확인 필요.
#
#     c.save()
#
#     watermark = PdfFileReader(open('watermark.pdf', 'rb'))
#
#     output_file = PdfFileWriter()
#
#     # 머지하기 위한 원본 PDF
#     input_file = PdfFileReader(open('test2.pdf', 'rb'))
#
#     page_count = input_file.getNumPages()
#     print('page count:' + str(page_count))
#
#     for page_number in range(page_count):
#         print("Watermarking page {} of {}".format(page_number + 1, page_count))
#         input_page = input_file.getPage(page_number)
#         input_page.mergePage(watermark.getPage(0))
#         output_file.addPage(input_page)
#
#     with open('result_output.pdf', 'wb') as outputStream:
#         output_file.write(outputStream)
#
# except Exception as err:
#     common.exception_print(err)


# PDF 텍스트 추출
# from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
# from pdfminer.converter import TextConverter
# from pdfminer.layout import LAParams
# from pdfminer.pdfpage import PDFPage
# from io import StringIO
#
#
# # 'D:/26.DataAnalisyPrj/ProxyServer/FastAPIServer/Data/Report/ML_Result'
# pdf_file_path = 'D:/26.DataAnalisyPrj/ProxyServer/FastAPIServer/Data/Report/ML_Result/MLReport.pdf'
#
#
# def convert_pdf_to_txt(path):
#     '''
#     PDF 파일 text 글자 추출
#     :param path: PDF 파일 경로
#     :return:
#     '''
#     rsrcmgr = PDFResourceManager()
#     retstr = StringIO()
#     codec = 'utf-8'
#     laparams = LAParams()
#     device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
#     fp = open(path, 'rb')
#     interpreter = PDFPageInterpreter(rsrcmgr, device)
#     password = ''
#     maxpages = 0
#     caching = True
#     pagenos = set()
#
#     for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,
#                                   caching=caching, check_extractable=True):
#         interpreter.process_page(page)
#         text = retstr.getvalue().replace('\n', '')
#
#     fp.close()
#     device.close()
#     retstr.close()
#     return text
#
#
# pdfFilePath = 'D:/8.병원별작업내용/_진료정보교류시스템_Ez/타사메뉴얼/진료정보교류_중외.pdf'
#
# extracted_text = convert_pdf_to_txt(pdfFilePath)
# print(extracted_text)


# PDF 문서 텍스트 작성
import CommonLib.Common as common
from fpdf import FPDF

try:
    width = 100
    height = 10
    centense_pos = 'C'

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for iCenterPos in range(1, 26):
        pdf.cell(width, height, txt='width:' + str(width) + " height:" + str(height) +
                                    " ln=" + str(iCenterPos) + " align=" + centense_pos,
                                    ln=iCenterPos, align=centense_pos)
        pdf.cell(width*2, height, txt='width:' + str(width*2) + " height:" + str(height) +
                                    " ln=" + str(iCenterPos) + " align=" + centense_pos,
                 ln=iCenterPos, align=centense_pos)
        pdf.cell(width, height, txt='width:' + str(width) + " height:" + str(height) +
                                    " ln=" + str(iCenterPos) + " align=L",
                                    ln=iCenterPos, align='L')
        pdf.cell(width, height, txt='width:' + str(width) + " height:" + str(height) +
                                    " ln=" + str(iCenterPos) + " align=R",
                                    ln=iCenterPos, align='R')
    pdf.output("test2.pdf", 'F')


except Exception as err:
    common.exception_print(err)