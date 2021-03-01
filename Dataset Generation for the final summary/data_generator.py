
import sys
import sqlite3 as sqlite
import csv
import random
from sklearn.model_selection import train_test_split
import argparse
import math
from collections import OrderedDict 
import statistics 
import numpy as np
import pandas as pd
import re


#params
cell_min_len = 20

test_set_size = .2




tablesToIgnore = ["sqlite_sequence"]

trainOutputFilename = "trainDataPoints.csv"
testOutputFilename = "testDataPoints.csv"


dataPointsArray = []


def ColEntropies(cursor, table, numRows, colNames):
    
    colEntropy = OrderedDict()
    for col in colNames:
        query = "SELECT COUNT(%s) FROM %s" %(col, table)
        cursor.execute(query)
        dat = cursor.fetchall()
        numUniqueValues = int(dat[0][0])

        bucket = {}
        for row in range(numRows):
            query = "SELECT %s FROM %s WHERE rowid = %d;" %(col, table, row+1)
            cursor.execute(query)
            dat = cursor.fetchall()
            dat = dat[0][0]
            if dat in bucket:
                bucket[dat] = bucket[dat]+1
            else:
                bucket[dat] = 1
        
        entropy = 0
        for entry in bucket:
            entropy = entropy + (-(bucket[entry]/numUniqueValues)*math.log2(bucket[entry]/numUniqueValues))
        colEntropy[col] = entropy
    return colEntropy


row_summary_templates = ["The %s is %s", 
            "%s, with a %s of %s", 
            "%s, and with a %s of %s", 
            "%s, and the %s is %s", 
            "%s, and %s is the %s"]
def createSentenceForRowSummary(tokens):
    
    #tokens will have length 2*l
    #l is the number of cols to read
    #hence the format is [<col,data>,<col,Data>,....]
    #this is flattened out to form [col,data,col,Data,....]


    txt = ""

    for i in range(0,len(tokens),2):
        col = tokens[i]
        val = tokens[i+1]

        if i == 0:
            #this is the first column this is been read
            txt = row_summary_templates[0] %(str(col), str(val))
        elif i == 2:
            txt = row_summary_templates[1] %(txt, str(col), str(val))
        else:
            idx = random.randint(2,4)
            if idx == 4:
                txt = row_summary_templates[idx] %(txt, val, col)
            else:
                txt = row_summary_templates[idx] %(txt, col, val)
    return txt

def colnameFilterForRowSummary(colname):
    k = str(colname)
    if "_" in str(k):
        temp = str.maketrans("_", " ") 
        k = str(k).translate(temp) 

    start = k.find( '(' )
    end = k.find( ')' )
    if start != -1 and end != -1:
        result = k[:start].strip()
        result1 = k[end+1:].strip()
        k = "%s %s" %(result, result1)
    return k

def DescribeTargetRowSummary(cursor, table, numRows, rowid, colNames, colEntropies, cell_data_threshold=200):

    #always read the first column
    #dont read column with low entropy value
    #dont read columns with hyphens, and does not apply to first column(for first column replace hyphen by space)
    #aggregate same values of two columns name:say like the two column names and their corresponding value
    #replace column name that has underscores to spaces and dont read data in parenthesis
    #if a cell has a lot of text we dont read it

    tokens = []

    colEntropies = colEntropies.copy()
    colEntropies.pop(colNames[0])
    colEntropies = dict(sorted(colEntropies.items(), key=lambda item: item[1], reverse=True))

    #returnText = None
    
    query = "SELECT %s FROM %s WHERE rowid = %d;" %(colNames[0], table, rowid+1)
    cursor.execute(query)
    data = cursor.fetchall()
    data = data[0][0]
    filteredColname = colnameFilterForRowSummary(colNames[0])
    tokens.append(filteredColname)
    tokens.append(data)
    #returnText = "Columns name %s, value %s" %(colNames[0], data)
    #appendText = ""
    count = int(len(colEntropies)/2)
    for k,v in colEntropies.items():
        count = count -1
        if count<0 :
            break
        query = "SELECT %s FROM %s WHERE rowid = %d;" %(k, table, rowid+1)
        cursor.execute(query)
        data = cursor.fetchall()
        data = data[0][0]
        if "-" in str(data) or len(str(data))>cell_data_threshold:
            continue

        k = colnameFilterForRowSummary(k)

        #appendText = appendText + (" , column name: %s, value: %s" %(k, data))
        tokens.append(k)
        tokens.append(data)
        
    if len(tokens)!=0:
        return createSentenceForRowSummary(tokens)

    else:
        return " no descriptions available for the record"

def DescribeTargetColumnSummary(cursor, table, col, numberOfRows, isColumnSummary, cell_data_threshold=200):
    
    #if column is textual and has low entropy Read out the probability distribution of these values
    #If probability < 10%  - assign it to ‘others’
    #If column is numerical and has high entropy,If small precision - Read number of unique values,If large precision - Read low, high and median values  
    #replace column name that has underscores to spaces and dont read data in parenthesis
    #if a cell has a lot of text we dont read it
    returnText = None
    colEntropy = OrderedDict() 
    diff_list = []
    columnName = col.replace('_', ' ')
    
    query = "SELECT %s FROM %s" %(col, table)
    cursor.execute(query)
    dat = cursor.fetchall()
    if isColumnSummary:
        returnText = "Column %s has %s entries with " %(columnName, str(len(dat)))
    else:
        returnText = "Column %s has entries with " %(columnName)
   
    
    empDfObj = pd.DataFrame(dat)
    unique_elements, counts_elements = np.unique(dat, return_counts=True)
    
    # If column is of datatype text and has less entropy
    if(empDfObj.dtypes == np.object).all():
        newColValues = [re.sub("[:\-() ]"," ",x) for x in list(sum(dat, ()))]
        if len(unique_elements) < 5:
            probability = [[x, round(y / sum(counts_elements) * 100)] for x, y in zip(unique_elements, counts_elements)]
            probabilityList = np.array(probability)
            probabilityList = probabilityList[probabilityList[:,-1].argsort()]
            for i,j in probabilityList[::-1][:3]:
                returnText = returnText + str(j) + " percent probability of " + i + " " 
            if len(probabilityList) > 3:
                returnText = returnText + " and rest of other values"
        else:
            # If others then read out the values
            returnText = returnText + " uniques value between " + newColValues[0] + " and " + newColValues[-1]
                
            
            
    # If column is of integers or float
    diff_list = list(sum(dat, ()))
    if (empDfObj.dtypes == np.float64).all() or (empDfObj.dtypes == np.int64).all():
        #If precison is more
        if max(np.diff(diff_list)) > 5:
            returnText = returnText + str(max(diff_list)) + " as maximum value  "+ str(statistics.median(diff_list)) + " as median and " + str(min(diff_list)) + " as minimum value" 
        else:
            returnText = returnText + " unique values ranging between  " + str(max(diff_list)) + " and " + str(min(diff_list))
            

    return returnText

def getStatForColumnSummary(cursor, table, col, numberOfRows, cell_data_threshold=200):

    #if column is textual and has low entropy Read out the probability distribution of these values
    #If probability < 10%  - assign it to ‘others’
    #If column is numerical and has high entropy,If small precision - Read number of unique values,If large precision - Read low, high and median values  
    #replace column name that has underscores to spaces and dont read data in parenthesis
    #if a cell has a lot of text we dont read it
    
    xCoordinate = ""
  
    colEntropy = OrderedDict() 
    diff_list = []
    
    query = "SELECT %s FROM %s" %(col, table)
    cursor.execute(query)
    dat = cursor.fetchall()
    
    empDfObj = pd.DataFrame(dat)
    unique_elements, counts_elements = np.unique(dat, return_counts=True)
    diff_list = list(sum(dat, ()))

    # If column is of datatype text and has less entropy
    if(empDfObj.dtypes == np.object).all():
        newColValues = [re.sub("[:\-() ]"," ",x) for x in list(sum(dat, ()))]
        if len(unique_elements) < 5:
            probability = [[x, round(y / sum(counts_elements) * 100)] for x, y in zip(unique_elements, counts_elements)]
            probabilityList = np.array(probability)
            probabilityList = probabilityList[probabilityList[:,-1].argsort()]
            #xCoordinate = xCoordinate + "::STAT"
            for i,j in probabilityList[::-1][:3]:
                #xCoordinate = xCoordinate + ";" + str(j) + ";" + i  
                xCoordinate = xCoordinate + str(j) + ":" + i+ ";"  
            
            
    # If column is of integers or float
    if (empDfObj.dtypes == np.float64).all() or (empDfObj.dtypes == np.int64).all():
        #If precison is more
        if max(np.diff(diff_list)) > 5:
            xCoordinate = xCoordinate + "min:" + str(min(diff_list)) + ";median:" + str(statistics.median(diff_list)) + ";max:" + str(max(diff_list)) + ";"
            #xCoordinate = xCoordinate + "::STAT" + ";min;" + str(min(diff_list)) + ";median;" + str(statistics.median(diff_list)) + ";max;" + str(max(diff_list))
            
            
    return xCoordinate

def selectColumnsForTableSummary(cursor, table, numRows, colNames, colEntropies):

    selectedColumns = []

    selectedColumns.append(colNames[0])

    colEntropies = colEntropies.copy()
    colEntropies.pop(colNames[0])
    colEntropies = dict(sorted(colEntropies.items(), key=lambda item: item[1], reverse=True))

    min = -1
    minList = []
    max = -1
    maxList = []

    for k,v in colEntropies.items():

        if min == -1 and max == -1:
            #initilly this condition will be called
            min = v
            minList.append(k)
            max = v
            maxList.append(k)
            continue

        if min>v:
            #if we have a element with a even smaller value
            min = v
            minList.clear()
            minList.append(k)
        elif min == v:
            #if we have same value as minimum then we have analyse which is the left most
            minList.append(k)

        if max<v:
            #if we have a element with a even larger value
            max = v
            maxList.clear()
            maxList.append(k)
        elif max == v:
            #if we have same value as maximum then we have analyse which is the left most
            maxList.append(k)

    minCol = None
    if len(minList)>1:
        #we have multiple elements in the minlist hence take the one in the left direction
        #colnames are in left to right order preseved
        for col in minList:
            if minCol is None:
                minCol = col
                continue
            for colItr in colNames:
                if minCol == colItr:
                    #if this is hit first, then this is left most
                    break
                elif col == colItr:
                    minCol = col
                    break
    else:
        minCol = minList[0]

    if minCol is not None:
        selectedColumns.append(minCol)

    maxCol = None
    if len(maxList)>1:
        #we have multiple elements in the minlist hence take the one in the left direction
        #colnames are in left to right order preseved
        for col in maxList:
            if col == minCol:
                #we use one column only once
                #hence if the column is same as mincol, skip it
                continue
            if maxCol is None:
                maxCol = col
                continue
            for colItr in colNames:
                if maxCol == colItr:
                    #if this is hit first, then this is left most
                    break
                elif col == colItr:
                    maxCol = col
                    break
    else:
        maxCol = maxList[0]

    if maxCol is not None:
        selectedColumns.append(maxCol)

    return selectedColumns

def DescribeTargetTableSummary(cursor, table, numberOfRows, columnsToDescribeForTableSummary):
    filteredtable = str(table).replace('_', ' ')
    returnText = "Table %s has %s rows "%(filteredtable, str(numberOfRows))
    if len(columnsToDescribeForTableSummary) > 0:
        for i in columnsToDescribeForTableSummary:
            returnText = returnText + " and " + DescribeTargetColumnSummary(cursor, table, i, numberOfRows, False)
    return returnText

def EncodeDataForCellSummary(cursor, table, rowid, colname):
    cellTemplate = "%s:%s;"
    query = "SELECT %s FROM %s WHERE rowid = %d;" %(colname, table, rowid+1)
    cursor.execute(query)
    data = cursor.fetchall()
    encoding = cellTemplate %(colname, str(data[0][0]))
    return encoding

def EncodeDataForRowSummary(cursor, table, rowid, colNames, colEntropies):

    encoding = None

    rowTemplate = "%s:%s;"
    rowTemplateFor2ndCellOnwards = "%s%s:%s;"

    colEntropies = colEntropies.copy()
    colEntropies.pop(colNames[0])
    colEntropies = dict(sorted(colEntropies.items(), key=lambda item: item[1], reverse=True))

    #returnText = None
    
    query = "SELECT %s FROM %s WHERE rowid = %d;" %(colNames[0], table, rowid+1)
    cursor.execute(query)
    data = cursor.fetchall()

    #filteredColname = colnameFilterForRowSummary(colNames[0])

    encoding = rowTemplate %(colNames[0], str(data[0][0]))

    
    for k,v in colEntropies.items():

        query = "SELECT %s FROM %s WHERE rowid = %d;" %(k, table, rowid+1)
        cursor.execute(query)
        data = cursor.fetchall()
        encoding = rowTemplateFor2ndCellOnwards %(encoding, k, str(data[0][0]))
    
    return encoding

def EncodeDataForTableSummary(cursor, table, numRows, colNames, colEntropies, columnsForTableSummary):
    encoding = None
    rowTemplate = "%s:%s;"
    rowTemplateFor2ndCellOnwards = "%s%s:%s;"

    colEntropies = colEntropies.copy()
    for j in range(len(columnsForTableSummary)):
        colEntropies.pop(columnsForTableSummary[j])
    colEntropies = dict(sorted(colEntropies.items(), key=lambda item: item[1], reverse=True))

    for i in range(numRows):
        
        for j in range(len(columnsForTableSummary)):

            query = "SELECT %s FROM %s WHERE rowid = %d;" %(columnsForTableSummary[j], table, i+1)
            cursor.execute(query)
            data = cursor.fetchall()
            if encoding is None:
                encoding = rowTemplate %(columnsForTableSummary[j], str(data[0][0]))
            else:
                encoding = rowTemplateFor2ndCellOnwards %(encoding, columnsForTableSummary[j], str(data[0][0]))

        for k,v in colEntropies.items():

            query = "SELECT %s FROM %s WHERE rowid = %d;" %(k, table, i+1)
            cursor.execute(query)
            data = cursor.fetchall()
            encoding = rowTemplateFor2ndCellOnwards %(encoding, k, str(data[0][0]))
        
        encoding = encoding+";"
        
    return encoding

def EncodeDataForColumnSummary(cursor, table, numberOfRows, col):
    
    
    xCoordinate = None
    diff_list = []
    
    query = "SELECT %s FROM %s" %(col, table)
    cursor.execute(query)
    dat = cursor.fetchall()
    empDfObj = pd.DataFrame(dat)
    unique_elements, counts_elements = np.unique(dat, return_counts=True)
    diff_list = list(sum(dat, ()))
    
    # Generating X Coordinate
    converted_list = [str(element) for element in diff_list]
    xCoordinate = col + ":" + ";".join(converted_list)
    
    return xCoordinate



def doProcessingForDB(cursor, table):
    try:
        print("started processing table "+table)
    
        if (table in tablesToIgnore):
            print("finished processing table "+table)
            return            
            
        columnsQuery = "PRAGMA table_info(%s)" % table
        cursor.execute(columnsQuery)
        tableMeta = cursor.fetchall()
        numberOfColumns = len(tableMeta)
        
        colNames = []
        for i in range(numberOfColumns):
            colNames.append(tableMeta[i][1])
        
        rowsQuery = "SELECT Count() FROM %s" % table
        cursor.execute(rowsQuery)
        numberOfRows = cursor.fetchone()[0]
        
        if numberOfRows == 0:
            print("no rows for table "+table)
            print("finished processing table "+table)
            return 

        isColumnSummary = False

        #required for row summary and choosing columns for table summary
        colEntropies = ColEntropies(cursor, table, numberOfRows, colNames )
        
        #case of table summary
        columnsToDescribeForTableSummary = selectColumnsForTableSummary(cursor, table, numberOfRows, colNames, colEntropies)
        tableSummary  = DescribeTargetTableSummary(cursor, table, numberOfRows, columnsToDescribeForTableSummary) 
        encodedData = EncodeDataForTableSummary(cursor, table, numberOfRows, colNames, colEntropies, columnsToDescribeForTableSummary)
        dat = [table,0,-1,-1, encodedData,"", tableSummary]
        #print(str(dat[0])+"\t"+str(dat[1])+"\t"+str(dat[2])+"\t"+str(dat[3]))
        dataPointsArray.append(dat)
        
        
        #case of row summary
        for i in range(numberOfRows):
            data = DescribeTargetRowSummary(cursor, table, numberOfRows, i, colNames, colEntropies)
            encodedData = EncodeDataForRowSummary(cursor, table, i, colNames, colEntropies)
            dat = [table,1,-1,i, encodedData,"", data]
            #print(str(dat[0])+"\t"+str(dat[1])+"\t"+str(dat[2])+"\t"+str(dat[3]))
            dataPointsArray.append(dat)
            
        #case of column summary
        for i in range(numberOfColumns):
            isColumnSummary = True
            data = DescribeTargetColumnSummary(cursor, table, colNames[i], numberOfRows, isColumnSummary)
            #dPoints = DescribeTargetColumnSummary(cursor, table, colNames[i], numberOfRows, isColumnSummary)
            encodedData = EncodeDataForColumnSummary(cursor, table, numberOfRows, colNames[i])
            colStat = getStatForColumnSummary(cursor, table, colNames[i], numberOfRows)
            dat = [table,2,i,-1, encodedData, colStat, data]
            #print(str(dat[0])+"\t"+str(dat[1])+"\t"+str(dat[2])+"\t"+str(dat[3]))
            dataPointsArray.append(dat)
            
        #case of cell summary
        for i in range(numberOfColumns):
            for j in range(numberOfRows):
             
                #sanity check
                #data has to be of len 200 or more
                query = "SELECT %s FROM %s WHERE rowid = %d;" %(colNames[i], table, j+1)
                cursor.execute(query)
                data = cursor.fetchall()
                if len(str(data[0][0]))<cell_min_len:
                    continue

                encodedData = EncodeDataForCellSummary(cursor, table, j, colNames[i])

                #the cell data is the target again
                dat = [table,3,i,j, encodedData, "", str(data[0][0])]
                #print(str(dat[0])+"\t"+str(dat[1])+"\t"+str(dat[2])+"\t"+str(dat[3]))
                dataPointsArray.append(dat)
                
        print("finished processing table "+table)
    except Exception as e:
        print("error processing table "+table)
        print(str(e))

def Describe(dbFile, examples_to_generate):
    connection = sqlite.connect(dbFile)
    cursor = connection.cursor()
    
    
    # Get List of Tables:      
    tableListQuery = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY Name"
    cursor.execute(tableListQuery)
    tables = map(lambda t: t[0], cursor.fetchall())
    tables = list(tables)
    
    
    for table in tables:
        doProcessingForDB(cursor, table)
        
    cursor.close()
    connection.close()  

    totalDataPoints = len(dataPointsArray)

    
    if examples_to_generate == -1:
        examples_to_generate = totalDataPoints
    elif totalDataPoints<examples_to_generate:
        print("There are at most "+str(totalDataPoints)+" number of examples.")
        examples_to_generate = totalDataPoints
        
        
    random.shuffle(dataPointsArray)
    requiredAmountData = dataPointsArray[:examples_to_generate]
    
    print("doing train test split")
    trainDataPointsArray ,testDataPointsArray = train_test_split(requiredAmountData,test_size=test_set_size)
    print(len(dataPointsArray))
    print(len(requiredAmountData))
    print(len(trainDataPointsArray))
    print(len(testDataPointsArray))
    
    return trainDataPointsArray ,testDataPointsArray 
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--numexamples", "-n", help="number of examples to generate")
    parser.add_argument("--testsplitsize", "-t", help="test split size")
    parser.add_argument("--cellminlen", "-c", help="minimum length of cell data")
    
    args = parser.parse_args()
        
    cell_min_len = int(args.cellminlen)
    
    test_set_size = float(args.testsplitsize)
    
    numExamplesToGenerate = int(args.numexamples)
    
    print(cell_min_len)
    print(test_set_size)
    print(numExamplesToGenerate)
    
    

    dbFile = r"D:\dke\3rd_SEM\dbse project\dataset genration\data\wikisql.db"
    trainDataPointsArray ,testDataPointsArray  = Describe(dbFile, numExamplesToGenerate)
    
    print("writing to file")
    
    with open(trainOutputFilename, 'w', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["tname", "qType", "col" ,"row", "data", "stat", "target"])
        writer.writerows(trainDataPointsArray)
        
    with open(testOutputFilename, 'w', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["tname", "qType", "col" ,"row", "data", "stat", "target"])
        writer.writerows(testDataPointsArray)

        