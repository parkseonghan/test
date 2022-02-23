import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import pandas as pd
import numpy as np
import CommonLib.Common as common

from matplotlib.colors import LinearSegmentedColormap
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

common.DataFrame_PrintFull(pd)


# 참고 블로그
# https://thenewth.wordpress.com/2020/09/15/%EC%97%B0%EA%B4%80-%EA%B7%9C%EC%B9%99-%EB%B6%84%EC%84%9Dfeat-python/


def Apriori_Run(request_json):
    try:
        if __name__ == "__main__":
            store_df = pd.read_csv(common.get_local_file_path() + '연관분석_사망환자_주부상병_2년자료_New.csv', header=None)
        else:
            store_df = pd.DataFrame(request_json)

        support_val = 0.005

        # 아이템을 리스트로 묶고 다시 리스트로 묶는다. -> 리스트의 리스트
        # ex) [['A318-1', 'J189', 'R53-4'], ['S0620-1', 'S0650']]
        records = []
        for i in range(len(store_df)):
            records.append([str(store_df.values[i, j])
                            for j in range(len(store_df.columns)) if not pd.isna(store_df.values[i, j])])

        # print(records)
        # print('='*100)

        # 학습시작
        te = TransactionEncoder()
        # fit:고유 라벨값 생성, transform:리스트를 원-핫 인코딩
        te_ary = te.fit(records).transform(records, sparse=True)

        # 결과 DataFrame으로 변환 (각항목 별 유무 True, False)
        te_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
        # print(te_ary)
        # print('=' * 100)

        # 각 아이템별 지지도 계산
        frequent_itemset = apriori(te_df,
                                   min_support=support_val,  # 최소 지지도 (0~1 사이의 숫자)
                                   max_len=4,  # 품목 집합 최대 개수
                                   use_colnames=True,  # 품목집합을 칼럼 이름으로 설정
                                   verbose=1)  # 진행상황 보기

        frequent_itemset['item_count'] = frequent_itemset['itemsets'].map(lambda x: len(x))
        frequent_itemset.sort_values('support', ascending=False, inplace=True)
        # print(frequent_itemset)
        # print('=' * 100)

        # 결과 support 항목이 min_threshold 지정된 값 이상만 출력
        # output
        # antecedents, consequents, antecedent, support, consequent, support, confidence, lift, leverage, conviction
        association_rules_df = association_rules(frequent_itemset,
                                                 # metric='confidence',
                                                 metric='support',
                                                 min_threshold=support_val)

        # print(association_rules_df)
        # print(100*"=")

        all_confidences = []
        collective_strengths = []
        cosine_similarities = []

        for _, row in association_rules_df.iterrows():

            all_confidence_if = list(row['antecedents'])[0]
            all_confidence_then = list(row['consequents'])[0]

            # print('antecedents:' + list(row['antecedents'])[0] + '    ' + str(row['antecedent support']))
            # print('consequents:' + list(row['consequents'])[0] + '    ' + str(row['consequent support']))

            # 지지도가 높은 아이템이 A로 선정
            if row['antecedent support'] <= row['consequent support']:
                all_confidence_if = list(row['consequents'])[0]
                all_confidence_then = list(row['antecedents'])[0]

            # 지지도 수치가 높은값을 선정
            all_confidence = {all_confidence_if + ' => ' +
                              all_confidence_then: row['support'] / max(row['antecedent support'],
                                                                        row['consequent support'])}
            all_confidences.append(all_confidence)

            violation = row['antecedent support'] + row['consequent support'] - 2 * row['support']
            ex_violation = 1 - row['antecedent support'] * row['consequent support'] - \
                           (1 - row['antecedent support']) * (1 - row['consequent support'])

            if violation == 0:
                violation = 0.000001
                collective_strength = (1 - violation) / (1 - ex_violation) * (ex_violation / violation)
                collective_strengths.append(collective_strength)
            else:
                collective_strength = (1 - violation) / (1 - ex_violation) * (ex_violation / violation)
                collective_strengths.append(collective_strength)

            # 코사인 유사도 (cosine_similarity)
            # 값과 관계 없이 벡터의 형식이 동일할 경우 1 https://wikidocs.net/24603
            cosine_similarity = row['support'] / np.sqrt(row['antecedent support'] * row['consequent support'])
            cosine_similarities.append(cosine_similarity)
            # print("유사도:" + str(cosine_similarity))
            # print(100 * '=')

        association_rules_df['all-confidence'] = all_confidences
        association_rules_df['collective strength'] = collective_strengths
        association_rules_df['cosine similarity'] = cosine_similarities
        # print(association_rules_df.head())

        topArray = []
        max_i = 4
        for i, row in association_rules_df.iterrows():
            '''
            지지도(support) : 한 거래 항목 안에 A와 B를 동시에 포함하는 거래의 비율. 지지도는 A와 B가 함계 등장할 확률이다.
                            전체 거래의 수를 A와 B가 동시에 포함된 거래수를 나눠주면 구할 수 있다.
            신뢰도(confidence) : 항목 A가 포함하는 거래에 A와 B가 같이 포함될 확률. 신뢰도는 조건부 확률과 유사하다.
                                A가 일어났을때 B의 확률이다. A의 확률을 A와 B가 동시에 포함될 확률을 나눠주면 구할 수 있다.
            향상도(lift) : A가 주어지지 않을 때의 품목 B의 확률에 비해 A가 주어졌을 때 품목 B의 증가비율. B의 확률이 A가 일어났을 때
                          B의 확률을 나눴을 때 구할 수 있다. lift값은 1이면 서로 독립적인 관계이며 1보다 크면 두 폼목이 서로 양의 상관 관계, 
                          1보다 작으면 두 품몸이 서로 음의 상관관계이다. A와 B가 독립이면 분모, 분자가 같기 때문에 1이 나온다.
            
            lift = 1, 품목간의 관계 없다.
            lift > 1, 품목간의 긍정적인
            *예: 우연히, 같이 사게되는 경우 관계가 있다.
            *예: 같이 사는 경우
            lift < 1, 품목간의 부정적인 관계가 있다.
            *예: 같이 사지 않는 경우                                                                       
            '''
            result_val = list(row['antecedents'])[0] + " => " + list(row['consequents'])[0] + "\n"
            result_val += "[Support : " + str(round(row['support'], 2)) + "]" + "\n"
            result_val += "[Confidence : " + str(round(row['confidence'], 2)) + "]" + "\n"
            result_val += "[Lift : " + str(round(row['lift'], 2)) + "]" + "\n"
            topArray.append(result_val)

            print("Rule: " + list(row['antecedents'])[0] + " => " + list(row['consequents'])[0])
            print("지지도(Support): " + str(round(row['support'], 2)))
            print("신뢰도(Confidence): " + str(round(row['confidence'], 2)))
            print("향상도(Lift): " + str(round(row['lift'], 2)))
            print("=====================================")

            if i == max_i:
                break

        # 시각화출력
        support_x = association_rules_df['support']  # 지지도 X축
        confidence_y = association_rules_df['confidence']  # 신뢰도 Y축

        # print("support_x")
        # print(support_x)
        # print(100*'=')
        # print("confidence_y")
        # print(confidence_y)
        # print(100*'=')

        h = 347
        # s = 1
        v = 1

        colors = [
            mcl.hsv_to_rgb((h / 360, 0.2, v)),
            mcl.hsv_to_rgb((h / 360, 0.55, v)),
            mcl.hsv_to_rgb((h / 360, 1, v))]

        cmap = LinearSegmentedColormap.from_list('my_cmap', colors, gamma=2)

        #           향상도    영향력      확신           지지도             집합력                  코사인 유사도
        measures = ['lift', 'leverage', 'conviction', 'all-confidence', 'collective strength', 'cosine similarity']

        if __name__ == "__main__":
            fig = plt.figure(figsize=(15, 10))
            fig.set_facecolor('white')

            for i, measure in enumerate(measures):
                ax = fig.add_subplot(320 + i + 1)
                if measure != 'all-confidence':
                    scatter = ax.scatter(support_x, confidence_y, c=association_rules_df[measure], cmap=cmap)
                else:
                    scatter = ax.scatter(support_x, confidence_y,
                                         c=association_rules_df['all-confidence'].map(
                                             lambda x: [value for item, value in x.items()][0]),
                                         cmap=cmap)
                ax.set_xlabel('support')
                ax.set_ylabel('confidence')
                ax.set_title(measure)

                fig.colorbar(scatter, ax=ax)

            fig.subplots_adjust(wspace=0.2, hspace=0.5)
            plt.show()

        association_rules_df['all_confidence_val'] = \
            association_rules_df['all-confidence'].map(lambda x: [value for item, value in x.items()][0])

        association_rules_df['all_confidence_item'] = \
            association_rules_df['all-confidence'].map(lambda x: [item for item, value in x.items()][0])

        association_rules_df.columns = ['antecedents', 'consequents', 'antecedent_support', 'consequent_support',
                                        'support', 'confidence', 'lift', 'leverage', 'conviction', 'all_confidence',
                                        'collective_strength', 'cosine_similarity', 'all_confidence_val',
                                        'all_confidence_item']
        association_rules_df = pd.concat([association_rules_df, pd.DataFrame(topArray, columns=['SUMMARY'])], axis=1)
        # print(association_rules_df.head())
        return association_rules_df

    except Exception as err:
        common.exception_print(err)


if __name__ == "__main__":
    Apriori_Run(None)
