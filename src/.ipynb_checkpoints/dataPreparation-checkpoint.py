import pandas
import numpy as np

def dropUselessData(df):
    df = df.drop(['city_name','AVERAGE_WIND_SPEED', 'AVERAGE_PRECIPITATION'], axis = 1)
    return df

def treatOutliers(df):

    cols = ['AVERAGE_FREE_FLOW_SPEED','AVERAGE_TIME_DIFF','AVERAGE_FREE_FLOW_TIME','AVERAGE_TEMPERATURE','AVERAGE_ATMOSP_PRESSURE','AVERAGE_HUMIDITY']
    Q1 = df[cols].quantile(0.25) 
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1

    condition = ~((df[cols] < (Q1 - 1.5 * IQR)) | (df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)

    df = df[condition]  
    
    """    
    for (columnName, columnData) in df.iteritems():
        df[columnName] = np.where(df[columnName] < df[columnName].quantile(0.10), df[columnName].quantile(0.10), df[columnName])
        df[columnName] = np.where(df[columnName] > df[columnName].quantile(0.90), df[columnName].quantile(0.90), df[columnName])
    """
    return df


def treatDate(df):
    df['record_date'] = pandas.to_datetime(df['record_date'])
    #format='%Y-%m-%d %H:%M:%S'
    df['day'] = df.record_date.dt.weekday
    df['month'] = df.record_date.dt.month
    df['week'] = df.record_date.dt.weekofyear
    df['hour'] = df.record_date.dt.hour
    df = df.drop(['record_date'], axis = 1)
    #print(df.head())
    return df


#Conversão de métricas - record_date, AVERAGE_SPEED_DIFF, LUMINOSITY, AVERAGE_CLOUDINESS, AVERAGE_RAIN
def convert(df): 
    #df.record_date = pandas.Categorical(df.record_date)
    #df['record_date'] = df.record_date.cat.codes

    if('AVERAGE_SPEED_DIFF' in df.columns):
        df.AVERAGE_SPEED_DIFF = pandas.Categorical(df.AVERAGE_SPEED_DIFF)
        df['AVERAGE_SPEED_DIFF'] = df.AVERAGE_SPEED_DIFF.cat.codes

    df.LUMINOSITY = pandas.Categorical(df.LUMINOSITY)
    df['LUMINOSITY'] = df.LUMINOSITY.cat.codes

    df.AVERAGE_CLOUDINESS = pandas.Categorical(df.AVERAGE_CLOUDINESS)
    df['AVERAGE_CLOUDINESS'] = df.AVERAGE_CLOUDINESS.cat.codes

    df.AVERAGE_RAIN = pandas.Categorical(df.AVERAGE_RAIN)
    df['AVERAGE_RAIN'] = df.AVERAGE_RAIN.cat.codes
    return df

def dataTreatment(df):
    df = convert(df)
    df = df.drop_duplicates()
    df = dropUselessData(df)
    #df = treatOutliers(df)
    df = treatDate(df) #Falta acabar
    return df

def testTreatment(df):
    df = convert(df)
    df = dropUselessData(df)
    df = treatDate(df) #Falta acabar
    return df
