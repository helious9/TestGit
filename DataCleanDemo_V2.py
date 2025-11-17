# ------------------------------------------------------------
#Author：JasonZhang
#Email：zhijian_zhang@ggec.com.cn

# 1.Copy CSV files（in RawData folder） and script in same folder
# 2.Run DataCleanDemo.py script
# Function：
# 1.Read all CSV files  this folder
# 2.Only keep one piece of data every minute, delete the rest, and get 720 rows of data
# 3.Draw  samples CPU Temperature average and difference graphs,save in \Processed_Files_Thermal_Average_Difference\
# ------------------------------------------------------------
import csv
import os
import shutil
import schedule
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
########################################################################
def task1():

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


    ########################################################################
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
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')], 
                      key=lambda x: int(''.join(filter(str.isdigit, x))))
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


    # 指定根文件夹路径
    root_folder = 'D:\WashData\Script'
    process_all_csv_files(root_folder)
    ########################################################################
    # Get header
    header=[]
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
                temp_average21 = subset.iloc[:, TraversalHeader].mean()

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
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv') and os.path.isfile(os.path.join(folder_path,f))], 
                      key=lambda x: int(''.join(filter(str.isdigit, x))))

    for file in csv_files:
        file_path = os.path.join(folder_path,file)

        df = pd.read_csv(file_path)

    #Find TimeStamp
        if df.columns[0]!='Timestamp':
           df.columns = ['Timestamp'] + df.columns[1:].tolist()


        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    #reverse
        df = df.sort_values(by='Timestamp',ascending=False)

    #Get 720 Line data
        df = df.head(1440)

        df.to_csv(file_path,index=False)

    print("Saved Lastest 12Hours Data")
    ##########################################################################

    ##########################################################################

    # Get header line keywords
    # header = []
    # with open(file_path, 'r') as file:
    #     reader = csv.reader(file)
    #     header = next(reader)
    #     index = header.index('CPU_T') #Traversal header line from 'CPU_T' to final
    #     TraversalHeader = index
    for num in range(len(header)-index): #debuging
        now = datetime.now()
        time_12_hours_ago = now - timedelta(hours=12)
        folder_name_A = time_12_hours_ago.strftime("%Y_%m%d_%H：%M")
        folder_name_B = now.strftime("%Y_%m%d_%H：%M")
        folder_name = f"{folder_name_A} —— {folder_name_B}"
        output_folder = folder_name
        os.makedirs(output_folder, exist_ok=True)
        csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')], 
                          key=lambda x: int(''.join(filter(str.isdigit, x))))

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
            
            # 获取实际文件编号（从文件名中提取数字）
            file_num = ''.join(filter(str.isdigit, file))

            # If the current file has lesser rows than the maximum time column, add NaN
            # cpu_t_data = pd.Series(df.iloc[:, 21] if len(df.columns) > 21 else [pd.NA] * max_rows)
            cpu_t_data = pd.Series(df.iloc[:, TraversalHeader] if len(df.columns) > TraversalHeader else [pd.NA] * max_rows)
            cpu_t_data = cpu_t_data.reindex(range(max_rows), fill_value=pd.NA)

            # Print
            print(f"File: {file}, DataLine: {len(cpu_t_data)}")
            print(df.iloc[0][21])  # Debuging line
            result_df[f'{header[TraversalHeader]}_{file_num}'] = cpu_t_data

        # Calculate the average and add to column [:, 1:CSV_Num + 1]
        result_df[f'Average_{header[TraversalHeader]}'] = result_df.iloc[:, 1:CSV_Num + 1].mean(axis=1)

        # Calculate the difference and add to column [:,CSV_Num+2:2*CSV_Num+1]
        for i, file in enumerate(csv_files, 1):
            file_num = ''.join(filter(str.isdigit, file))
            # result_df[f'Diff_{header[TraversalHeader]}_{i}'] = result_df[f'{header[TraversalHeader]}_{i}'] - result_df['Average_{header[TraversalHeader]}']
            result_df[f'Dif_{header[TraversalHeader]}_{file_num}'] = (result_df[f'{header[TraversalHeader]}_{file_num}'] -
                                                                    result_df[f'Average_{header[TraversalHeader]}'])

        # Save Date To New CSV
        output_file_path = os.path.join(output_folder, f'{header[TraversalHeader]}.csv')
        result_df.to_csv(output_file_path, index=False)
        print(f"Cleaned，Data Saved in '{output_file_path}'")


        # Merge two diagrams into one
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  #OK
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # Draw the first picture: column 2 to column CSV_Num+1
        for col in result_df.columns[1:CSV_Num + 1]:
            file_num = col.split('_')[-1]  # 获取文件名中的数字部分
            ax1.plot(result_df['TimeStamp'], result_df[col], label=f'{header[TraversalHeader]}_{file_num}')

        ax1.set_xlabel('TimeStamp')
        ax1.set_ylabel('Values')
        ax1.set_title(f'{header[TraversalHeader]}_Average')
        ax1.legend()

        # Time form
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
        ax1.tick_params(axis='x', rotation=45)  # rotate timestamp

        # # Draw the second picture: column CSV_Num+2 to final
        for col in result_df.columns[CSV_Num + 1:]:
            if col.startswith(f'Dif_{header[TraversalHeader]}'):
                file_num = col.split('_')[-1]  # 获取文件名中的数字部分
                ax2.plot(result_df['TimeStamp'], result_df[col], label=f'Dif_{file_num}')
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

####################################################################################
task1()
# schedule.every().day.at("11:05") .do(task1)
# # schedule.every().day.at("19:46") .do(task1)
# while True:
#     schedule.run_pending()