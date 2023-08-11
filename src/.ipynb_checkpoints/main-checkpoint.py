import dataTraining as dt
import dataVisualization as dv
import dataPreparation as dp
import sys
import pandas

TRAINING_DATA_PATH = '../datasets/training_data.csv'
TEST_DATA_PATH = '../datasets/test_data.csv'
SUBMISSION_FILE = '../datasets/submission.csv'

def writeSubmissionFile(model, df_test):
    predicts = model.predict(df_test)
    with open(SUBMISSION_FILE, 'w') as file:
        i = 1
        file.write('RowId,Speed_Diff\n')

        for elem in predicts:
            if elem == 0:
                token = 'None'
            elif elem == 1:
                token = 'Low'
            elif elem == 2:
                token = 'Medium'
            elif elem == 3:
                token = 'High'
            else:
                token = 'Very_High' 
            file.write(str(i) + ',' + token + '\n')
            i += 1
    '''
    with open(SUBMISSION_FILE, 'w') as file:
        i = 1
        file.write('RowId,Speed_Diff\n')

        for elem in results:
            if elem == 0:
                token = 'None'
            elif elem == 1:
                token = 'Low'
            elif elem == 2:
                token = 'Medium'
            elif elem == 3:
                token = 'High'
            else:
                token = 'Very_High' 
            file.write(str(i) + ',' + token + '\n')
            i += 1
    '''

def main():
    if(len(sys.argv) <= 1):
        print('''
Nenhum modelo selecionado:
    1 -> Decision Tree Classifier
    2 -> Decision Tree Regressor
    3 -> Linear Regression
    4 -> Logistic Regression
    5 -> Suport Vector Machine
    6 -> Grid Search
''')
        quit()

    # Read datasets
    df_training = pandas.read_csv(TRAINING_DATA_PATH)
    df_test = pandas.read_csv(TEST_DATA_PATH)
    
    #Prepare data
    df_training = dp.dataTreatment(df_training)
    df_test = dp.testTreatment(df_test)
    
    # Visualize data
    #dv.dataVisualization(df_training) 
    
    #Train Data
    model = dt.dataTraining(int(sys.argv[1]), df_training) 
    
    #Write to submission file
     
    if model != []:
        writeSubmissionFile(model, df_test)
        print("Data training completed... Submission file is ready.")
    else:
        print("Data training failed")
 
if __name__ == "__main__":
    main()
