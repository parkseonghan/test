import pandas as pd
import CommonLib.Common as common

common.DataFrame_PrintFull(pd)


def Correlation_run(request_json):

    try:

        col_name1 = 'x'
        col_name2 = 'y'
        col_name3 = 'corr'

        if __name__ == "__main__":
            df = pd.read_csv(common.get_local_file_path() + '상관분석_혈압나이성별온도.csv')
        else:
            df = pd.DataFrame(request_json)

        df = df.dropna(axis=0)
        df = df.astype('float')

        # 상관관계 매트릭스
        corr_df = df.corr()
        corr_df = corr_df.apply(lambda x: round(x, 2))
        corr_df2 = corr_df.reset_index()
        corr_df_unstack = corr_df.unstack()

        # 상관관계가 높은 순서대로
        descending = pd.DataFrame(corr_df_unstack[corr_df_unstack <= 1].sort_values(ascending=False), columns=['corr'])
        descending = descending.reset_index()
        descending.columns = [col_name1, col_name2, col_name3]
        # print(corr_df2)
        # ax = sns.heatmap(corr_df, annot= True, cmap= 'Blues')
        # plt.show()

        # sns.pairplot(corr_df)
        # plt.show()

        return corr_df2.copy()

    except Exception as err:
        common.exception_print(err)


if __name__ == "__main__":
    Correlation_run(None)
