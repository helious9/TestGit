'''
Auther：Jason zhang
#Email：zhijian_zhang@ggec.com.cn
# 1.Copy CSV files（in RawData folder） and script in same folder
# 2.Run DataCleanDemo.py script
# Function：
# 1.Clip data to new folder
# 2.Only keep one piece of data every minute, delete the rest, and get 1440 rows of data
# 3.Draw  samples CPU Temperature average and difference graphs,save in output folder
'''

import os
import csv
import re
import schedule
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# config path
# parent_folder = "your_parent_folder"  # FAther Folder path
parent_folder = "D:/HWV/edmt/EDMT_Daily/2025-11-15/EDMT/ADMT-14-11"  # write your father folder path
# Clip_time_to_folder = "D:/WashData/Script/EDMT_GroupC"
# yesterday
# YESTERDAY = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

# output
# Clip_time_to_folder = f"D:/WashData/Script/EDMT_GroupC/{YESTERDAY}"
Clip_time_to_folder = f"D:/HWV/edmt/EDMT_Daily/2025-11-15/EDMT/ADMT-14-11/Output_graph"
def task1():
    YESTERDAY = (datetime.today() - timedelta(days=269)).strftime('%Y-%m-%d')
    # YESTERDAY = (datetime.today()).strftime('%Y-%m-%d')

    # traversal all folder ，keywords like "uutlog"
    for sub_folder in os.listdir(parent_folder):
        if "uutlog" in sub_folder:  # select folder
            folder_path = os.path.join(parent_folder, sub_folder)
            # output folder path

            # create yesterday file
            # folder_name = f"{YESTERDAY}"
            # Clip_time_to_folder = folder_name
            # os.makedirs(Clip_time_to_folder, exist_ok=True)
            if not os.path.isdir(folder_path):  # ensure
                continue

            # Get folder num， Num
            match = re.search(r'\d+', sub_folder)
            Num = match.group() if match else "Unknown"
            print(f"Processing folder: {sub_folder}, Num = {Num}")

            # select CSV files
            csv_files = [f for f in os.listdir(folder_path) if "usage_data.csv" in f and f.endswith(".csv")]

            # save data
            combined_df = pd.DataFrame()
            header_row = None

            for idx, file in enumerate(csv_files):
                file_path = os.path.join(folder_path, file)
                # change timestamp col format  #debugging 2025/3/9
                # read CSV
                df = pd.read_csv(file_path)
                # change timestamp col format  #debugging 2025/3/9
                if 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    print("Warning：DataFrame not find 'Timestamp' column")

                # "Time" column named "Timestamp"（if exist）
                if "Time" in df.columns and "Timestamp" not in df.columns:
                    df.rename(columns={"Time": "Timestamp"}, inplace=True)

                # only save one header line
                if header_row is None:
                    header_row = df.columns.tolist()

                # delete  column2 ~ 21column （Python index start 0）
                # df.drop(df.columns[1:21], axis=1, inplace=True)

                # append DataFrame
                combined_df = pd.concat([combined_df, df])

            # "Timestamp"  reverse CSV
            if "Timestamp" in combined_df.columns:
                combined_df.sort_values(by="Timestamp", ascending=False, inplace=True)
            # "Get yesterday line" debugging 2025/3/8
            if 'Timestamp' in combined_df.columns:
                # 将 Timestamp 列转换为日期时间类型
                combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
                # 筛选出 2025 年 2 月 20 日和 2025 年 2 月 21 日的数据
                # filtered_df = df[df['Timestamp'].dt.strftime('%Y/%m/%d').isin(['2025/02/20', '2025/02/21'])]
                filtered_df = combined_df[combined_df['Timestamp'].dt.strftime('%Y-%m-%d').isin([YESTERDAY])] # OK 2025/3/8
                # filtered_df = combined_df[combined_df['Timestamp'].dt.strftime('%Y-%m-%d').isin(['2025/2/11'])] # OK 2025/3/8
                combined_df = filtered_df  # debugging 2025/3/8
                # 将筛选后的数据保存回原文件
                # filtered_df.to_csv(file_path, index=False)
                print(f"已处理文件: {idx}")
            else:
                print(f"文件 {idx} 中没有 'Timestamp' 列，跳过处理。")
            # Debugging line 2025/3/8
            # save files
            output_file = os.path.join(Clip_time_to_folder, f"merged_usage_data_{Num}.csv")
            # combined_df.to_csv(output_file, index=False)
            combined_df.to_csv(output_file, index=False)
            print(f"处理完成，合并后的文件保存在: {output_file}")
            print(f"标题行: {header_row}")
            print(YESTERDAY)


    #   Delete useless CSV
    def delete_small_csv_files_in_current_folder(size_threshold=2048):
        """
        删除当前文件夹中小于 size_threshold 字节（默认 2KB）的 CSV 文件。
        :param size_threshold: 文件大小阈值，单位为字节
        """
        current_folder = os.getcwd()
        deleted_files = 0

        for filename in os.listdir(current_folder):
            file_path = os.path.join(current_folder, filename)
            if filename.endswith(".csv") and os.path.isfile(file_path):
                if os.path.getsize(file_path) < size_threshold:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                    deleted_files += 1

        print(f"删除完成，共删除 {deleted_files} 个文件。")

    delete_small_csv_files_in_current_folder()

    def clean_units_in_csv(folder_path):
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                filepath = os.path.join(folder_path, filename)

                # 读取CSV文件内容
                with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = [row for row in reader]

                # 如果文件有数据行
                if rows:
                    # 查找要删除的列索引
                    headers = rows[0]
                    columns_to_delete = [i for i, header in enumerate(headers)
                                         if header == "PLAY_STATUS"]

                    # 处理单位文本
                    modified = False
                    for row in rows:
                        for i in range(len(row)):
                            original = row[i]
                            # 使用正则表达式移除"mV"、"mA"和"%"
                            cleaned = re.sub(r'(\d+)(m[VA]|%|dK)', r'\1', original)
                            if cleaned != original:
                                row[i] = cleaned
                                modified = True

                    # 删除指定列（从后往前删除以避免索引变化）
                    if columns_to_delete:
                        for col_index in sorted(columns_to_delete, reverse=True):
                            for row in rows:
                                if col_index < len(row):
                                    del row[col_index]
                        modified = True
                        print(f'已删除"{filename}"中的"PLAY_STATUS"列')

                    # 如果有修改，则写回文件
                    if modified:
                        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(rows)
                        print(f'已处理文件: {filename}')
                    else:
                        print(f'无需处理文件: {filename}')
                else:
                    print(f'空文件跳过: {filename}')

    # folder_path = input('Please input your path: ')
    # clean_units_in_csv(folder_path)
    clean_units_in_csv(Clip_time_to_folder)
###########################################################################
    def delete_play_status_column(csv_path):
        try:
            # Read CSV File
            df = pd.read_csv(csv_path)

            # Check "PLAY_STATUS" Column exist or not
            if "PLAY_STATUS" in df.columns:
                # Delete "PLAY_STATUS" Column
                df = df.drop(columns=["PLAY_STATUS"])
                # Save modified data to CSV file
                df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"处理文件 {csv_path} 时出现错误: {e}")

    ########################################################################
    ########################################################################
    def delete_play_status_column(csv_path):
        try:
            # 读取 CSV 文件
            df = pd.read_csv(csv_path)

            # 检查是否存在 "PLAY_STATUS" 列
            if "PLAY_STATUS" in df.columns:
                # 删除 "PLAY_STATUS" 列
                df = df.drop(columns=["PLAY_STATUS"])
                # 将修改后的数据保存回原文件
                df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"处理文件 {csv_path} 时出现错误: {e}")


    def process_all_csv_files(root_folder):
        # 遍历指定文件夹及其子文件夹
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith('.csv'):
                    # 构建完整的 CSV 文件路径
                    csv_path = os.path.join(root, file)
                    # 调用删除列的函数
                    delete_play_status_column(csv_path)


    # 指定根文件夹路径，你可以根据实际情况修改
    root_folder = 'D:\HWV\edmt\EDMT_Daily/2025-11-15\EDMT'
    process_all_csv_files(root_folder)
    ########################################################################
    # def CopySN(sn_txt_file,sn_csv_file):
    # # 定义输入的txt文件路径和输出的csv文件路径
    #     # 打开txt文件和csv文件
    #     with open(sn_txt_file, 'r', encoding='utf-8') as txt_f, open(sn_csv_file, 'w', newline='', encoding='utf-8') as csv_f:
    #         # 创建csv写入对象
    #         csv_writer = csv.writer(csv_f)
    #
    #         # 逐行读取txt文件
    #         for line in txt_f:
    #             # 按空格分割每行的数据
    #             data = line.strip().split()
    #             # 将分割后的数据写入csv文件
    #             csv_writer.writerow(data)
    #
    #     print(f"数据已成功从 {sn_txt_file} 转存到 {sn_csv_file}")
    #     # open MojaveSN.csv
    #     df = pd.read_csv(sn_csv_file)
    #     # if df.columns[0:1]!='interface':
    #     #      df.columns = ['interface'] + df.columns[1:].tolist()
    #     SN = df.loc[:,'interface']
    #     return SN
    #     # df['interface'] = pd.to_datetime(df['interface'])
    #     # SN = df
    #
    #     # print (SN[0])
    # sn_txt_file = r'C:\edmt2\uutinfo_ed.txt'
    # sn_csv_file = r'D:\WashData\Configuration\MojaveSN.csv'
    # SN = CopySN(sn_txt_file,sn_csv_file)
    # print(SN[0])
    ########################################################################
    current_file_path = __file__
    current_file_dir = os.path.dirname(current_file_path)
    folder_path = current_file_dir


    # Cleaned_folder = os.path.join(folder_path, "Cleaned_Data")
    Cleaned_folder = os.path.dirname(current_file_path)
    # Ensure folder exist
    if not os.path.exists(Cleaned_folder):
        os.makedirs(Cleaned_folder)

    # traversal
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    Num_csv = len(csv_files)

    # delete useless column
    def delete_play_status_column(csv_path):
        try:
            # 读取 CSV 文件
            df = pd.read_csv(csv_path)

            # 检查是否存在 "PLAY_STATUS" 列
            if "PLAY_STATUS" in df.columns:
                # 删除 "PLAY_STATUS" 列
                df = df.drop(columns=["PLAY_STATUS"])
                # 将修改后的数据保存回原文件
                df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"处理文件 {csv_path} 时出现错误: {e}")


    def process_all_csv_files(root_folder):
        # 遍历指定文件夹及其子文件夹
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith('.csv'):
                    # 构建完整的 CSV 文件路径
                    csv_path = os.path.join(root, file)
                    # 调用删除列的函数
                    delete_play_status_column(csv_path)


    # 指定根文件夹路径，你可以根据实际情况修改
    root_folder = 'D:\HWV\edmt\EDMT_Daily/2025-11-15\EDMT'
    process_all_csv_files(root_folder)
    ########################################################################
    # Get header
    header=[]
    print(f"当前目录: {folder_path}")
    print(f"找到的CSV文件列表: {csv_files} ")
    with open(csv_files[0], 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        index = header.index('CCG6_CHRG_T') #Traversal header line from 'CPU_T' to final
        TraversalHeader = index
    # traversal
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        time_col = pd.to_datetime(df.iloc[:, 0])
        data_col = df.iloc[:, 21]  # locate data

        # Accurate to the minute, minus the second
        time_col_minute = time_col.dt.floor('min')

        # Save Data
        cleaned_data = []

        # traversal timestamp
        for time in time_col_minute.unique():

            subset = df[time_col_minute == time]

            # get average

            for num in range(len(header)-index):
                numeric_data = pd.to_numeric(subset.iloc[:, TraversalHeader],errors='coerce')
                temp_average21 = numeric_data.mean(skipna=True)

                if pd.isna(temp_average21):
                    temp_average21 = 0

                # place average to
                subset.iloc[0, TraversalHeader] = temp_average21
                TraversalHeader+=1
            TraversalHeader=index
            cleaned_data.append(subset.iloc[0])


        cleaned_df = pd.DataFrame(cleaned_data)

        # SaveCSV
        # output_path = os.path.join(Cleaned_folder, f"Cleaned_{csv_file}")
        output_path = os.path.join(Cleaned_folder, f"{csv_file}")
        cleaned_df.to_csv(output_path, index=False)

        print(f"file {csv_file} Cleaned，Saved as {output_path}")

    ##########################################################################
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and os.path.isfile(os.path.join(folder_path,f))]

    for file in csv_files:
        file_path = os.path.join(folder_path,file)

        df = pd.read_csv(file_path)

    #Find TimeStamp
        if df.columns[0]!='Timestamp':
           df.columns = ['Timestamp'] + df.columns[1:].tolist()


        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    #reverse
        df = df.sort_values(by='Timestamp',ascending=False)

    #Get 720Line data
        df = df.head(1440)

        df.to_csv(file_path,index=False)

    print("Saved Lastest 12Hours Data")
    ##########################################################################

    ##########################################################################
    for num in range(len(header)-index): #debuging
        now = datetime.now()
        time_12_hours_ago = now - timedelta(hours=24)
        folder_name_A = time_12_hours_ago.strftime("%Y_%m%d_%H") #"%Y_%m%d_%H_%M"
        folder_name_B = now.strftime("%Y_%m%d_%H")
        folder_name = f"{folder_name_A} —— {folder_name_B}"
        output_folder = folder_name
        os.makedirs(output_folder, exist_ok=True)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        #len
        CSV_Num = len(csv_files)
        # Traversal header line from 'CPU_T' to final
        # TraversalHeader = index
        # Find Longest timestamp
        longest_timestamp = None
        max_rows = 0

        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            # file_path = os.path.join(folder_path, file) OK
            df = pd.read_csv(file_path)

            print(f"file: {file}, row: {len(df)}")

            if len(df) >max_rows:
                max_rows = len(df)
                longest_timestamp = pd.to_datetime(df.iloc[:, 0])

        # Create NewDataFrame Temp
        result_df = pd.DataFrame()
        result_df['TimeStamp'] = longest_timestamp

        # Add all CSV file 22Column 'CPU_T' To DataFrame
        # header[21]——header[len(header[])-1]
        for i, file in enumerate(csv_files):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            # If the current file has lesser rows than the maximum time column, add NaN
            # cpu_t_data = pd.Series(df.iloc[:, 21] if len(df.columns) > 21 else [pd.NA] * max_rows)
            cpu_t_data = pd.Series(df.iloc[:, TraversalHeader] if len(df.columns) > TraversalHeader else [pd.NA] * max_rows)
            cpu_t_data = cpu_t_data.reindex(range(max_rows), fill_value=pd.NA)

            # Print
            print(f"File: {file}, DataLine: {len(cpu_t_data)}")
            print(df.iloc[0][21])  # Debugging line
            result_df[f'{header[TraversalHeader]}_{i + 1}'] = cpu_t_data

        # Calculate the average and add to column [:, 1:CSV_Num + 1]
        result_df[f'Average_{header[TraversalHeader]}'] = result_df.iloc[:, 1:CSV_Num + 1].mean(axis=1)

        # Calculate the difference and add to column [:,CSV_Num+2:2*CSV_Num+1]
        for i in range(1, CSV_Num + 1):
            # result_df[f'Diff_{header[TraversalHeader]}_{i}'] = result_df[f'{header[TraversalHeader]}_{i}'] - result_df['Average_{header[TraversalHeader]}']
            result_df[f'Dif_{header[TraversalHeader]}_{i}'] = (result_df[f'{header[TraversalHeader]}_{i}'] -
                                                                result_df[f'Average_{header[TraversalHeader]}'])

        # Save Date To New CSV
        output_file_path = os.path.join(output_folder, f'{header[TraversalHeader]}.csv')
        result_df.to_csv(output_file_path, index=False)
        print(f"Cleaned，Data Saved in '{output_file_path}'")


        # Merge two diagrams into one
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  #OK
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # Draw the first picture: column 2 to column CSV_Num+1
        for col in result_df.columns[1:CSV_Num + 2]:
            ax1.plot(result_df['TimeStamp'], result_df[col], label=col)

        ax1.set_xlabel('TimeStamp')
        ax1.set_ylabel('Values')
        ax1.set_title(f'{header[TraversalHeader]}_Average')
        ax1.legend()

        # Time form
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
        ax1.tick_params(axis='x', rotation=45)  # rotate timestamp

        # # Draw the second picture: column CSV_Num+2 to final
        for col in result_df.columns[CSV_Num + 2:]:
            ax2.plot(result_df['TimeStamp'], result_df[col], label=col)
        ax2.set_xlabel('TimeStamp')
        ax2.set_ylabel('Difference Values')
        ax2.set_title(f'{header[TraversalHeader]}_Differences')
        ax2.legend()

        # Time form
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
        ax2.tick_params(axis='x', rotation=45)  # rotate timestamp
        plt.tight_layout()
        print(index)

        #Save

        # output_combined_image_path = os.path.join(output_folder, 'Combined_CPU_T.png')
        output_combined_image_path = os.path.join(output_folder, f'Combined_{header[TraversalHeader]}')
        plt.savefig(output_combined_image_path)

        TraversalHeader += 1
        print(TraversalHeader)
    ################################################################################
    print(header[21]) #ok
    print(index)
    print(len(header))


###########################################################################
task1()
# # schedule.every().day.at("19:45") .do(task1)
# schedule.every().day.at("10:58") .do(task1)
# while True:
#     schedule.run_pending()
