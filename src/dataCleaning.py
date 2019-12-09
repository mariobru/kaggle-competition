#!/usr/bin/env python3 
import pandas as pd

def dfCleaning(dataframe):
    color = {
        "D": 7,
        "E": 6,
        "F": 5,
        "G": 4,
        "H": 3,
        "I": 2,
        "J": 1
    }

    cut = {
        "Ideal": 5,
        "Premium": 4,
        "Very Good": 3,
        "Good": 2,
        "Fair":1
    }

    clarity = {
        "IF": 8,
        "VVS1": 7,
        "VVS2": 6,
        "VS1": 5,
        "VS2": 4,
        "SI1": 3,
        "SI2": 2,
        "I1": 1
    }

    for i in range(len(dataframe)):
        co = color[dataframe.loc[i, "color"]]
        cu = cut[dataframe.loc[i, "cut"]]
        cl = clarity[dataframe.loc[i, "clarity"]]
        dataframe.loc[i, "color_num"] = co
        dataframe.loc[i, "cut_num"] = cu
        dataframe.loc[i, "clarity_num"] = cl

    dataframe.drop(columns=['cut','color','clarity'], inplace=True)
    dataframe['volume'] = dataframe['x']*dataframe['y']*dataframe['z']
    dataframe.drop(['x','y','z'],axis=1,inplace=True)

    return dataframe
