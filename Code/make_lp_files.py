import numpy as np
import pandas as pd
import os


# Create a directory to save the .lp files
directory = "linear_prog_files"
if not os.path.exists(directory):
    os.mkdir(directory)
    print ("Folder: " , directory, " has been created to store the .lp files")
else:
    print ("Folder for storing the .lp file already exists")

# Read the data from the 'User & Task ID' spreadshit of the 'COMP3217CW2Input.xlsx' and put format is an NumPy array
users = np.asarray(pd.read_excel('COMP3217CW2Input.xlsx', header = None, sheet_name ='User & Task ID'))
users = users[1:]   # Discard the title row

# Read the data from 'TestingResults.txt' and put format is an NumPy array
prices = np.asarray(pd.read_csv('TestingResults.txt', sep = ',', header = None))


# Loop through each row of the guideline price curves predicted in the 'TestingResults.txt' file
for row in range (len(prices)):
   # If it is labelled as an abnormal guideline price curve start creating the linear programming content
    if prices[row][24] == 1: 
        c = "c="            # Create the objective function to be solved
        constraints = []    # List containing the energy constraints for the tasks of each user
        boundaries = []     # List containing the energy boundaries for every task of each user
        
        # Loop through the users and their tasks
        for i in range (len(users)):
            user_num = (users[i][0])[4]
            task_num = (users[i][0])[10:]  
            
            energy_str = ""     # Energy constraint string
            
            # Loop through the ready time and the deadline time to add to the objective function (c)  
            for time in range (users[i][1], users[i][2]+1):
                task = "x" + str(task_num) + "_" + str(time)
                c = c + str(prices[row][time]) + " " + task + "+"
                
                
                energy_str = energy_str + task + "+"            # Create the energy constraints
                boundaries.append('0<=' + str(task) + '<=1;')   # Create the energy boundaries
            
            # Add the two strings to the arrays
            energy_str = energy_str[:-1] + '=' + str(users[i][4]) + ';'
            constraints.append(energy_str)
           
            #next user is reached. Write the contents in a file
            if(i%10 == 9):
                c = c[:-1] + ';'    #add ; in the end of the objective function
                #write the contents in an .lp file
                file = "linear_prog_files/user_" + str(user_num) + "_guideline_" + str(row) + ".lp"
                file = open(file, 'w')

                file.write('/* Objective function */'+ '\n' + 'min: c;')
                file.write('\n' + '\n')
                file.write(c + '\n')
                
                for y in (constraints):
                    file.write(y+'\n')
                
                for k in range (len(boundaries)):
                    file.write(boundaries[k]+'\n')
                
                file.write('\n')
                file.write('/* Variable bounds */')
                file.close()



                # Make the objective function, the contraints and the boundaries ready for the next user
                c = "c="
                constraints.clear()
                boundaries.clear()
