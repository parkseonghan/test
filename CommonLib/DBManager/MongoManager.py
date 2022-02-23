import CommonLib.Common as common
from pymongo import MongoClient


class MongoDB_Con:
    def __init__(self, dbname, ip, port):
        self.DBName = dbname
        self.Ip = ip
        self.Port = port
        self.Client = MongoClient("mongodb://" + self.Ip + ":" + self.Port + "/")
        self.DB = self.Client[self.DBName]
        print(self.DB.list_collection_names())

    def Update(self, collection, data, updateData, multiple=False):
        try:
            if multiple:
                self.DB[collection].update_many(data, updateData)
            else:
                print(collection, data, updateData)
                self.DB[collection].update_one(data, updateData)
        except Exception as err:
            common.exception_print(err)

    def Insert(self, collection, data):
        self.DB[collection].insert_one(data)

    def Select_find_one(self, collection, data, where):
        return self.DB[collection].find_one(data)[where]

    def Select_find(self, collection, data):
        select_arr = []
        for item in self.DB[collection].find(data):
            select_arr.append(item)
        return select_arr


def Insert_by_DataFrame(collection, df):
    try:
        list = []
        for index in range(0, len(df)):
            list.append(index)

        df['db_index'] = list
        df['db_index'] = df['db_index'].astype('str')
        df = df.set_index('db_index')
        obj = MongoDB_Con(common.MongoDBName, common.MongoDBIP, common.MongoDBPort)
        obj.Insert(collection, df.to_dict())

    except Exception as err:
        common.exception_print(err)


def Insert_by_Dictionary(collection, dic):
    try:
        db = MongoDB_Con(common.MongoDBName, common.MongoDBIP, common.MongoDBPort)
        db.Insert(collection, dic)

    except Exception as err:
        common.exception_print(err)


def Update_One(collection, data, update_data):
    try:
        db = MongoDB_Con(common.MongoDBName, common.MongoDBIP, common.MongoDBPort)
        update_data = {"$set":update_data}
        db.Update(collection, data, update_data)

    except Exception as err:
        common.exception_print(err)


if __name__ == "__main__":
    data = {"Report_File":"MLReport_Prophet_20220210140738558845"}
    update_data = {"ImageYN":"N"}
    Update_One('Prophet', data, update_data)
