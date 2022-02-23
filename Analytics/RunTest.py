
query_temp = """
                                            SELECT %s, COUNT(*) AS VAL
                                              FROM %s
                                             WHERE 1 = 1
                                               AND %s >= '%s'
                                               AND %s <= '%s'
                                          GROUP BY %s
                                          ORDER BY %s ASC """
ColName = 'AccountDate'
print(query_temp % (ColName, 'Pdo_Master', ColName, '2021-01-01', ColName, '2021-01-10', ColName, ColName))