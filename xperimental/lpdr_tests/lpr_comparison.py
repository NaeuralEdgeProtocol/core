import pandas as pd
import numpy as np


def convert_reading(description):
  if 'control' in description:
    return description
  else:
    return 'camera'+description


if __name__ == '__main__':
  csv_path = r'C:\Users\bleot\Downloads\LPR 12- 19.01.xlsx - Evenimente Unitati.csv'
  df = pd.read_csv(csv_path)
  print(df.head())

