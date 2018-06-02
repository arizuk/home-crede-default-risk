import pandas as pd
import src.utils as utils

def build_last_application():
    prev = utils.read_csv('./input/previous_application.csv')
    last = prev.sort_values(['DAYS_DECISION'], ascending=False).groupby(['SK_ID_CURR']).nth(0)
    last.to_pickle('./features/last_application.pkl')
    print('Build ./features/last_application.pkl')

if __name__ == '__main__':
    build_last_application()