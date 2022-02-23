from collections import Counter
from konlpy.tag import Okt

import CommonLib.Common as common
import pytagcloud
import webbrowser
import csv


def morphological_run(request_json):
    try:
        result_arr = []
        start_index = 0
        max_cnt = 20
        max_fontSize = 130
        img_hieght = 900
        img_width = 600
        if __name__ == "__main__":
            with open(common.get_local_file_path() + '형태소자료_CRM요청글_박성찬.csv', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    result_arr.append(row[start_index])
        else:
            with open(request_json) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    result_arr.append(row[start_index])

        okt = Okt()
        noun = okt.nouns(''.join(result_arr))  # 명사추출

        # 한글자 이하 제외
        for i, v in enumerate(noun):
            if len(v) < 2:
                noun.pop(i)

        count = Counter(noun)

        # 명사 빈도
        noun_list = count.most_common(max_cnt)
        # for text in noun_list:
        #     print(text)

        # tag에 color, size, tag 사전 구성
        word_count_list = pytagcloud.make_tags(noun_list, maxsize=max_fontSize)  # maxsize : 최대 글자크기
        pytagcloud.create_tag_image(word_count_list,
                                    'wordCloud_Result.jpg',
                                    size=(img_hieght, img_width),
                                    fontname='korean',
                                    rectangular=True)

        webbrowser.open('wordCloud_Result.jpg')

    except Exception as err:
        common.exception_print(err)


if __name__ == '__main__':
    morphological_run(None)
