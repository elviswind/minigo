import zipline
from minigo import wrapper, dao
import numpy as np
import pandas 

def initialize(context):
    pipe = zipline.pipeline.Pipeline()  
    pipe = zipline.api.attach_pipeline(pipe, name='my_pipeline')  
    dollar_volume = zipline.pipeline.factors.AverageDollarVolume(window_length=5)  
    pipe.add(dollar_volume, 'dollar_volume')  
    high_dollar_volume = dollar_volume.top(850)  
    pipe.set_screen(high_dollar_volume)  

    zipline.api.schedule_function(
        rebalance,
        zipline.api.date_rules.week_start(),
        zipline.api.time_rules.market_open(),
    )

def rebalance(context, data):
    results = zipline.api.pipeline_output('my_pipeline')  
    df = data.history(results.index, fields="price", bar_count=200, frequency="1d").resample('1W').last().T
    df = df.dropna(axis=0)
    df.to_csv('d2.csv')
    wrapper.run(2, 'd2.csv')
    winner = pandas.DataFrame(np.load('lasttime.npy', allow_pickle=True)).iloc[:100,:].sort_values(4, ascending=False).iloc[0][0]
    #winner = [26, 95, 172, 301, 303, 486, 510, 565, 749, 762]
    chosen = df.iloc[winner]
    p = dao.get_dpr(chosen)[1]
  
    for i in range(len(chosen.index)): 
        zipline.api.order_target_value(chosen.index[i], context.account.net_liquidation * p[i])

    pass