# df = pd.read_csv(common.get_local_file_path() + "다중회귀_혈압나이성별온도.csv")

import os
import shutil
import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import CommonLib.Common as common
from matplotlib import rcParams
from datetime import datetime
from fpdf import FPDF

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False


def generate_sales_data(month: int) -> pd.DataFrame:
    try:
        dates = pd.date_range(
            start=datetime(year=2020, month=month, day=1),
            end=datetime(year=2020, month=month, day=calendar.monthrange(2020, month)[1])
        )

        sales = np.random.randint(low=1000, high=2000, size=len(dates))

        return pd.DataFrame({
            'Date': dates,
            'ItemsSold': sales
        })
    except Exception as err:
        common.exception_print(err)


def plot(data: pd.DataFrame, filename: str) -> None:
    try:
        plt.figure(figsize=(12, 4))
        plt.grid(color='#F2F2F2', alpha=1, zorder=0)
        plt.plot(data['Date'], data['ItemsSold'], color='#087E8B', lw=3, zorder=5)
        plt.title(f'Sales 2020/{data["Date"].dt.month[0]}', fontsize=17)
        plt.xlabel('Period', fontsize=13)
        plt.xticks(fontsize=9)
        plt.ylabel('Number of items sold', fontsize=13)
        plt.yticks(fontsize=9)
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        return
    except Exception as err:
        common.exception_print(err)


def construct():
    try:
        # 폴더가 있으면 삭제 후 다시 생성
        shutil.rmtree(PLOT_DIR)
        os.mkdir(PLOT_DIR)
    except FileNotFoundError:
        os.mkdir(PLOT_DIR)

    try:
        # 2020년 1월을 제외한 모든 달에 대해서 반복
        for i in range(2, 13):
            # 시각화 이미지 저장
            plot(data=generate_sales_data(month=i), filename=f'{PLOT_DIR}/{i}.png')

        # 문서에 표시된 데이터 구성
        counter = 0
        pages_data = []
        temp = []

        # 모든 plot 로드
        files = os.listdir(PLOT_DIR)
        # 월별로 정렬
        files = sorted(os.listdir(PLOT_DIR), key=lambda x: int(x.split('.')[0]))

        for fname in files:
            if counter == 3:
                pages_data.append(temp)
                temp = []
                counter = 0

            temp.append(f'{PLOT_DIR}/{fname}')
            counter += 1

        return [*pages_data, temp]
    except Exception as err:
        common.exception_print(err)


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297

    def header(self):
        self.image('Analytics/newsCrawling.png', 10, 8, 33)
        self.set_font('Arial', 'B', 11)
        self.cell(self.WIDTH - 80)
        self.cell(60, 1, 'Sales report', 0, 0, 'R')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def page_body(self, images):
        if len(images) == 3:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
            self.image(images[2], 15, self.WIDTH / 2 + 90, self.WIDTH - 30)
        elif len(images) == 2:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
        else:
            self.image(images[0], 15, 25, self.WIDTH - 30)

    def print_page(self, images):
        self.add_page()
        self.page_body(images)


try:
    december = generate_sales_data(month=12)
    print(december)
    plot(data=december, filename='december.png')

    PLOT_DIR = 'plots'
    plots_per_page = construct()

    pdf = PDF()

    for elem in plots_per_page:
        pdf.print_page(elem)

    pdf.output('SalesRepot.pdf', 'F')
except Exception as err:
    common.exception_print(err)









##############################################################################
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import CommonLib.Common as common
#
# # 정규표현식
# import re
#
# # 한국어 형태소 분석 - 명사단위
# from konlpy.tag import Okt
# from collections import Counter
#
# # BoW 벡터 생성
# from sklearn.feature_extraction.text import CountVectorizer
#
# # TF-IDF 적용 : Bag of words 벡터에 대해서 TF-IDF 변환 진행합니다.
# from sklearn.feature_extraction.text import TfidfTransformer
#
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import confusion_matrix
#
# # 블로그 : https://hyemin-kim.github.io/2020/08/29/E-Python-TextMining-2/
#
#
# def text_cleaning(text):
#     hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')
#     result = hangul.sub('', text)
#     okt = Okt()
#     nouns = okt.nouns(result)
#     nouns = [x for x in nouns if len(x) > 1]          # 한글자 제외
#     nouns = [x for x in nouns if x not in stopwords]  # 불용어 제거
#     return nouns
#
#
# # 한글 추출 규칙: 띄어 쓰기(1 개)를 포함한 한글 (특수문자 제거)
# def apply_regular_expression(text):
#     hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')
#     result = hangul.sub('', text)
#     return result
#
#
# def rating_to_label(rating):
#     if rating > 3:
#         return 1
#     else:
#         return 0
#
#
# try:
#
#     df = pd.read_csv(common.get_local_file_path() + "형태소분석_간호기록.csv", encoding='cp949')
#     common.DataFrame_Information(df)
#     df = df.dropna()
#
#     # # Okt() : 명사 형태소 추출
#     # okt = Okt()
#     #
#     # # 말뭉치(corpus)에 적용해서 명사 형태소 추출
#     # corpus = "".join(df['text'].tolist())  # 리스트 값을 문자열로 변환
#     #
#     # # 전체 말뭉치에서 명사 형태추출 : ['여행', '집중', '휴식', '제공', '호텔'....
#     # nouns = okt.nouns(apply_regular_expression(corpus))
#     #
#     # # 빈도 탐색 : [('호텔', 803), ('수', 498), ('것', 436)...
#     # counter = Counter(nouns)
#     #
#     # # 한글자 형식의 명사를 제거
#     # available_counter = Counter({x: counter[x] for x in counter if len(x) > 1})
#     # available_counter.most_common(10)
#
#     # 우리, 매우 와 같은 불용어 제외
#     # RANKS NL에서 제공해주는 한국어 불용어 사전을 활용
#     stopwords = pd.read_csv(common.get_local_file_path() + "형태소분석_간호기록.csv", encoding='cp949').values.tolist()
#     stopwords[:10]
#
#     # 특정 불용어 추가
#     self_stopwords = ['제주', '제주도', '호텔', '리뷰', '숙소', '여행', '트립']
#     for word in self_stopwords:
#         stopwords.append(word)
#
#     # BoW 벡터 생성
#     vect = CountVectorizer(tokenizer=lambda x: text_cleaning(x))
#     bow_vect = vect.fit_transform(df['text'].tolist())
#     word_list = vect.get_feature_names()
#     count_list = bow_vect.toarray().sum(axis=0)
#
#     word_count_dict = dict(zip(word_list, count_list))
#     print(word_count_dict)
#
#     tfidf_vectorizer = TfidfTransformer()
#     tf_idf_vect = tfidf_vectorizer.fit_transform(bow_vect)
#     print(vect.vocabulary_)
#
#     invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}
#     print(invert_index_vectorizer)
#     df['y'] = df['rating'].apply(lambda x: rating_to_label(x))
#
#     ################################################################################
#
#     x = tf_idf_vect
#     y = df['y']
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
#
#     # 로지스틱 회귀
#     lr = LogisticRegression(random_state=0)
#     lr.fit(x_train, y_train)
#     y_pred = lr.predict(x_test)
#
#     print('accuracy: %.2f' % accuracy_score(y_test, y_pred))
#     print('precision: %.2f' % precision_score(y_test, y_pred))
#     print('recall: %.2f' % recall_score(y_test, y_pred))
#     print('F1: %.2f' % f1_score(y_test, y_pred))
#
#     # confu = confusion_matrix(y_true=y_test, y_pred=y_pred)
#     #
#     # plt.figure(figsize=(4, 3))
#     # sns.heatmap(confu, annot=True, annot_kws={'size': 15}, cmap='OrRd', fmt='.10g')
#     # plt.title('Confusion Matrix')
#     # plt.show()
#
#     # 샘플링 재조정
#     positive_random_idx = df[df['y'] == 1].sample(275, random_state=12).index.tolist()
#     negative_random_idx = df[df['y'] == 0].sample(275, random_state=12).index.tolist()
#
#     random_idx = positive_random_idx + negative_random_idx
#     x = tf_idf_vect[random_idx]
#     y = df['y'][random_idx]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
#
#     lr2 = LogisticRegression(random_state=0)
#     lr2.fit(x_train, y_train)
#     y_pred = lr2.predict(x_test)
#
#     print('accuracy: %.2f' % accuracy_score(y_test, y_pred))
#     print('precision: %.2f' % precision_score(y_test, y_pred))
#     print('recall: %.2f' % recall_score(y_test, y_pred))
#     print('F1: %.2f' % f1_score(y_test, y_pred))
#
#     confu = confusion_matrix(y_true=y_test, y_pred=y_pred)
#
#     plt.figure(figsize=(4, 3))
#     sns.heatmap(confu, annot=True, annot_kws={'size': 15}, cmap='OrRd', fmt='.10g')
#     plt.title('Confusion Matrix')
#     plt.show()
#
#     print(lr2.coef_)
#     plt.figure(figsize=(10, 8))
#     plt.bar(range(len(lr2.coef_[0])), lr2.coef_[0])
#     plt.show()
#
#     coef_pos_index = sorted(((value, index) for index, value in enumerate(lr2.coef_[0])), reverse=True)
#     coef_neg_index = sorted(((value, index) for index, value in enumerate(lr2.coef_[0])), reverse=False)
#
#     invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}
#     print(invert_index_vectorizer)
#
#     for coef in coef_pos_index[:20]:
#         print(invert_index_vectorizer[coef[1]], coef[0])
#
#
# except Exception as err:
#     common.exception_print(err)