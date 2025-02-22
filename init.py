#! python3
# -*- encoding: utf-8 -*-
'''
@File    : init.py
@Time    : 2025/02/18 17:36:13
@Author  : longfellow 
@Email   : longfellow.wang@gmail.com
@Version : 1.0
@Desc    : None
'''


# 初始化脚本

from db.migrate import reset_tables
from db.milvus import milvus_client

if __name__ == '__main__':
    '''文档数据库初始化'''

    # 提示用户确认是否继续执行
    user_input = input("""
    警告：此操作将删除所有数据表并重新创建，所有数据将被清空。
    是否要继续执行数据库初始化操作？输入 'y' 继续，其他任意键退出：""")
    if user_input.lower() != 'y':
        print("\n操作已取消。")
        exit(0)
    else:
        reset_tables()
            
        if milvus_client.has_collection("langchat"):
            milvus_client.detele_collection("langchat")

        milvus_client.create_collection("langchat", 1024)
