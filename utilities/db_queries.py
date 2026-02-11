import os
import sqlite3
import numpy as np
from bisect import bisect

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def query_datetimes_newer_than_sqlite(sqlite_db_path, datetime_str, table):
    connection = sqlite3.connect(sqlite_db_path)
    
    with connection:
        connection.row_factory = dict_factory
        cur = connection.cursor()    
        sql = f'SELECT * FROM {table} WHERE DateTime>"{datetime_str}";'
        result = cur.execute(sql).fetchall()
        return result
    
def query_datetimes_between_sqlite(sqlite_db_path, datetime_str1, datetime_str2, table):
    connection = sqlite3.connect(sqlite_db_path)
    
    with connection:
        connection.row_factory = dict_factory
        cur = connection.cursor()    
        sql = f'SELECT * FROM {table} WHERE DateTime>"{datetime_str1}" AND DateTime<"{datetime_str2}";'
        result = cur.execute(sql).fetchall()
        return result

def query_table_sqlite(sqlite_db_path, datetime_str, table):
    connection = sqlite3.connect(sqlite_db_path)
    
    with connection:
        connection.row_factory = dict_factory
        cur = connection.cursor()    
        sql = f'SELECT * FROM {table} WHERE DateTime="{datetime_str}";'
        result = cur.execute(sql).fetchall()
        return result

def query_table_sqlite_with_list(sqlite_db_path, datetime_str_list, table):
    connection = sqlite3.connect(sqlite_db_path)
    
    with connection:
        connection.row_factory = dict_factory
        cur = connection.cursor()    
        sql = f'SELECT * FROM {table} WHERE DateTime IN "{datetime_str_list}";'
        result = cur.execute(sql).fetchall()
        return result

def leftOrRightWhitelight(t ,t_prev, t_next):
    best = None

    if    ( t_prev is None)   and (not t_next is None):
        return t_next
    elif (not t_prev is None) and   (t_next is None)  :
        return t_prev

    else:
#         print(t_prev,t_next,t)
        a = int(os.path.basename(t_prev)[3:-4])
        b = int(os.path.basename(t_next)[3:-4])

        c = int(os.path.basename(t)[3:-4])

        # print( 'prev: ', a, ', next: ', b, ', current: ' , c)

        d1 = np.abs(c-a)
        d2 = np.abs(c-b)

        if d1 <= d2:
            best = t_prev
        else:
            best = t_next

    return best

def find_closest_wl(name, wl_list):
    idx = bisect(wl_list, name)

    a = None if idx < 1 else wl_list[idx-1]
    b = None if idx >= len(wl_list) else wl_list[idx]
    
    return leftOrRightWhitelight(name, a, b)