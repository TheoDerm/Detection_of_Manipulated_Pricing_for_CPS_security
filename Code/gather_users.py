import os
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt





# Function for creating the graphs for the combined data frame of users
def draw_plot(df,num):
    
    df['hour'] = df['task'].str.split("_").str[1]
    df['power'] = df['power'].astype(int)
    df['hour'] = df['hour'].astype(int)
    df['user'] = df['user'].astype(int)

    df2 = np.asarray(df)
    
    hours = np.zeros(24,dtype=int)
    x_label = [str(i) for i in range(0, 24)]
    
    for i in range(len(df2)):
        hour = df2[i,3]
        power = df2[i,1]
        hours[hour] = hours[hour] + power

    plt.bar(x_label,hours)
    plt.xlabel('Hour of the Day')
    plt.ylabel('Total Community Energy Usage')
    plt.title(num)
    
   # plt.show()
    plt.savefig('Graphs/guideline ' + num)
    plt.clf()





guidelines = []
directory = os.getcwd()     #directory of the file
joint_directory = os.path.join(directory,'linear_prog_files\solved_lp')     #directory of the solved .lp files

# Loop through the directory of the solved_lp files
for f in os.listdir(joint_directory):
    file_name = os.fsdecode(f)
    user = ''
    guide_number = -1 
    
    if(len(file_name)) == 23 :
        guide_number = file_name[17:19]
        user = file_name[6]
    elif (len(file_name) == 22):
        guide_number = file_name[17]
        user = file_name[6]

    # If there is not a folder created for this guideline, create one
    if (guide_number !=-1) and guide_number not in guidelines: 
        guidelines.append(guide_number)

        new_file = 'linear_prog_files\guideline_'+ guide_number
        if not os.path.exists(new_file):
            os.mkdir(new_file)
    

    # Copy the user's guideline file to the guideline's directory 
    if(guide_number != -1):
        file_path = os.path.join(joint_directory,file_name) #here
        shutil.copy(file_path, ('linear_prog_files\guideline_'+ guide_number))


# After gathering the users, create the combined dataframes to make the graphs
joint_directory = os.path.join(directory,'linear_prog_files')
for f in os.listdir(joint_directory):
    file_name = os.fsdecode(f)

    if (file_name[:9] == 'guideline'):
        
        guideline_directory = os.path.join(joint_directory,file_name)    #directory paht of the guideline
        dfs = []    # List for the combined dataframes
        file_number = file_name[10:]

        # Loop through the guideline directory and combine the dataframes
        for txt_file in os.listdir(guideline_directory): 
            user = txt_file[5]
            file_path = os.path.join(guideline_directory,txt_file)
            df = (pd.read_csv(file_path, sep ="\s+"))
            df = df.iloc[2:,:-3]
            df.rename({'Value': 'task', 'of': 'power'}, axis=1, inplace=True)
            df['user'] = user
            
            dfs.append(df)
            
        final_dfs = pd.concat(dfs,axis=0,ignore_index=True)    # Final combined dataframe for the whole directory
        draw_plot(final_dfs,file_number)
      

    
        


    
  