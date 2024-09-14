import psycopg2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus, PULP_CBC_CMD, GUROBI_CMD, GUROBI
import gurobipy
from sqlalchemy import create_engine

import os

CONNECTION = "postgres://leloq@127.0.0.1/intradaydb"
CONNECTION_ALCHEMY = "postgresql://leloq@127.0.0.1/intradaydb"
conn = psycopg2.connect(CONNECTION)
conn_alchemy = create_engine(CONNECTION_ALCHEMY)
cursor = conn.cursor()
cursor.execute("ROLLBACK")

def get_average_prices(side, execution_time_start, execution_time_end, end_date, min_trades=10):

    # set start_of_day to end_date minus 1 day
    start_of_day = pd.to_datetime(end_date) - pd.Timedelta(hours=2)

    # set hour and minute to 0 (europe/berlin time)
    start_of_day = start_of_day.replace(hour=0, minute=0)

    end_of_day = start_of_day

    end_of_day = end_of_day.replace(hour=23, minute=45)

    cursor.execute(f"""
        SELECT 
        deliverystart, 
        SUM(price*volume)/SUM(volume) AS weighted_avg_price
        FROM 
        transactions_intraday_de
        WHERE 
        (executiontime BETWEEN '{execution_time_start}' AND '{execution_time_end}')
        AND (product ='XBID_Quarter_Hour_Power' or product = 'Intraday_Quarter_Hour_Power') AND side='{side}' AND deliverystart < '{end_date}' AND deliverystart >= '{start_of_day}'
        GROUP BY 
        deliverystart
        HAVING 
        COUNT(*) >= {min_trades};
        """)
    result = cursor.fetchall()
    result = [(row[0], float(row[1])) for row in result] # transform to float from decimal

    
    df = pd.DataFrame(result, columns=['deliverystart', 'price'])

    df['deliverystart'] = pd.to_datetime(df['deliverystart']).dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
    df.set_index('deliverystart', inplace=True)

    date_range = pd.date_range(start_of_day, end_of_day, freq='15min', tz='Europe/Berlin')

    df = df.reindex(date_range)

    print(df)

    return df
def get_closest_prices(execution_time_start, end_date):

    # set start_of_day to end_date minus 1 day
    start_of_day = pd.to_datetime(end_date) - pd.Timedelta(hours=2)

    # set hour and minute to 0 (europe/berlin time)
    start_of_day = start_of_day.replace(hour=0, minute=0)

    end_of_day = start_of_day

    end_of_day = end_of_day.replace(hour=23, minute=45)

    # get list of all 15 minute intervals from start_of_day to end_of_day
    intervals = pd.date_range(start_of_day, end_of_day, freq='60min')

    intervals_list = "','".join(intervals.strftime('%Y-%m-%d %H:%M:%S%z'))

    print(intervals_list)

    cursor.execute(f"""
        WITH RankedPrices AS (
            SELECT 
                deliverystart,
                price,
                executiontime,
                ROW_NUMBER() OVER(PARTITION BY deliverystart ORDER BY ABS(EXTRACT(EPOCH FROM ('{execution_time_start}' - executiontime)))) as rank
            FROM 
                transactions_intraday_de
            WHERE 
                product IN ('XBID_Quarter_Hour_Power', 'Intraday_Quarter_Hour_Power')  -- List of products
                AND deliverystart IN ('{intervals_list}')
                AND deliverystart > '{execution_time_start}'
                AND ABS(EXTRACT(EPOCH FROM ('{execution_time_start}' - executiontime))) <= 150   -- Only prices with at most 1 hour difference
        )
        SELECT 
            deliverystart,
            price
        FROM 
            RankedPrices
        WHERE 
            rank = 1;

        """)
    
    
    result = cursor.fetchall()
    df = pd.DataFrame(result, columns=['product', 'price'])

    # set index to product
    df.set_index('product', inplace=True)

    # set index to be all 15 minute intervals from start_of_day to end_of_day, filling missing values with NaN
    df = df.reindex(pd.date_range(start_of_day, end_of_day, freq='60min'))

    return df

def get_prices_day(df, execution_time_start, day):

    # set start_of_day to day at 00:00:00
    start_of_day = pd.to_datetime(day)

    # set hour and minute to 0 (europe/berlin time)
    start_of_day = start_of_day.replace(hour=0, minute=0)

    end_of_day = start_of_day

    end_of_day = end_of_day.replace(hour=23, minute=45)

    filtered_df = df[df['execution_time_start'] == execution_time_start]

    # filter so product is <= end_of_day
    filtered_df = filtered_df[filtered_df['product'] <= end_of_day]

    # remove column execution_time_start
    filtered_df = filtered_df.drop(columns=['execution_time_start'])

    # set index to product
    filtered_df.set_index('product', inplace=True)

    # set index to be all 15 minute intervals from start_of_day to end_of_day, filling missing values with NaN
    filtered_df = filtered_df.reindex(pd.date_range(start_of_day, end_of_day, freq='15min'))

    return filtered_df

def calculate_discounted_price(price, current_time, delivery_time, discount_rate):

    time_difference = (delivery_time - current_time).total_seconds() / 3600  # difference in hours

    if time_difference <= 1:  # if less than one hour, return the original price
        return price

    if (price < 0):
        discount_factor = np.exp((discount_rate/100) * time_difference)
    else:
        discount_factor = np.exp(-(discount_rate/100) * time_difference)

    return price * discount_factor

def run_optimization_quarterhours_repositioning(prices_qh, execution_time, cap, c_rate, roundtrip_eff, max_cycles, threshold, threshold_abs_min, discount_rate, prev_net_trades=pd.DataFrame(columns=["sum_buy", "sum_sell", "net_buy", "net_sell", 'product'])):

    # copy prices_qh
    prices_qh_adj = prices_qh.copy()

    # loop through prices_qh and adjust prices
    for i in prices_qh_adj.index:
        if not pd.isna(prices_qh_adj.loc[i, 'price']):
            prices_qh_adj.loc[i, 'price'] = calculate_discounted_price(prices_qh_adj.loc[i, 'price'], execution_time, i, discount_rate)

            # round prices to 2 decimals
            prices_qh_adj.loc[i, 'price'] = round(prices_qh_adj.loc[i, 'price'], 2)

    # copy prices_qh
    prices_qh_adj_buy = prices_qh.copy()

    # loop through prices_qh and adjust prices
    for i in prices_qh_adj_buy.index:
        if not pd.isna(prices_qh_adj_buy.loc[i, 'price']):
            prices_qh_adj_buy.loc[i, 'price'] = calculate_discounted_price(prices_qh_adj_buy.loc[i, 'price'], execution_time, i, -discount_rate)

            # round prices to 2 decimals
            prices_qh_adj_buy.loc[i, 'price'] = round(prices_qh_adj_buy.loc[i, 'price'], 2)

    prices_qh['price'] = round(prices_qh['price'], 2)

    # # merhe prices_qh_adj to prices_qh with column name "price_adj"
    # prices_qh = pd.merge(prices_qh, prices_qh_adj, left_index=True, right_index=True, suffixes=('', '_adj'))

    # print(prices_qh)

    # Create the 'battery' model
    m_battery = LpProblem('battery', LpMaximize)

    # Create variables using the DataFrame's index
    current_buy_qh = LpVariable.dicts("current_buy_qh", prices_qh.index, lowBound=0)
    current_sell_qh = LpVariable.dicts("current_sell_qh", prices_qh.index, lowBound=0)
    battery_soc = LpVariable.dicts("battery_soc", prices_qh.index, lowBound=0)

    # Create net variables
    net_buy = LpVariable.dicts("net_buy", prices_qh.index, lowBound=0)
    net_sell = LpVariable.dicts("net_sell", prices_qh.index, lowBound=0)
    charge_sign = LpVariable.dicts("charge_sign", prices_qh.index, cat='Binary')

    # Introduce auxiliary variables
    z = LpVariable.dicts("z", prices_qh.index, lowBound=0)
    w = LpVariable.dicts("w", prices_qh.index, lowBound=0)

    M = 100 

    e = 0.01

    # Objective function
    # Adjusted objective component for cases where previous trades < e
    adjusted_obj = [((current_sell_qh[i] * (prices_qh_adj.loc[i,'price'] - max(abs((threshold/100) * abs(prices_qh.loc[i,'price'])), threshold_abs_min)/2 - e)) - (current_buy_qh[i] * (prices_qh_adj_buy.loc[i,'price'] + max(abs((threshold/100) * abs(prices_qh.loc[i,'price'])), threshold_abs_min)/2 + e))) * 1.0/4.0 for i in prices_qh.index if not pd.isna(prices_qh.loc[i, 'price'])  and (prev_net_trades.loc[i,'net_buy'] < e and prev_net_trades.loc[i,'net_sell'] < e)]

    # Original objective component for cases where previous trades >= e
    original_obj = [(current_sell_qh[i] * (prices_qh.loc[i,'price'] - e) - current_buy_qh[i] * prices_qh.loc[i,'price']) * 1.0/4.0 for i in prices_qh.index if not pd.isna(prices_qh.loc[i, 'price']) and (prev_net_trades.loc[i,'net_buy'] >= e or prev_net_trades.loc[i,'net_sell'] >= e)]

    # Combine and set the objective
    m_battery += lpSum(original_obj + adjusted_obj)
    

    # Constraints
    previous_index = prices_qh.index[0]

    efficiency = roundtrip_eff**0.5

    for i in prices_qh.index[1:]:
        m_battery += battery_soc[i] == battery_soc[previous_index] + net_buy[previous_index]*efficiency*1.0/4.0 - net_sell[previous_index]*1.0/4.0/efficiency, f"BatteryBalance_{i}"
        previous_index = i

    m_battery += battery_soc[prices_qh.index[0]] == 0, "InitialBatterySOC"

    for i in prices_qh.index:
        # Handling NaN values by setting buy and sell quantities to 0
        if pd.isna(prices_qh.loc[i, 'price']):
            m_battery += current_buy_qh[i] == 0, f"NaNBuy_{i}"
            m_battery += current_sell_qh[i] == 0, f"NaNSell_{i}"
        else:
            m_battery += battery_soc[i] <= cap, f"Cap_{i}"
            m_battery += net_buy[i] <= cap * c_rate, f"BuyRate_{i}"
            m_battery += net_sell[i] <= cap * c_rate, f"SellRate_{i}"
            m_battery += net_sell[i]*1.0/efficiency/4.0 <= battery_soc[i], f"SellVsSOC_{i}"

        # big M constraints for net buy and sell
        m_battery += net_buy[i] <= M * charge_sign[i], f"NetBuyBigM_{i}"
        m_battery += net_sell[i] <= M * (1-charge_sign[i]), f"NetSellBigM_{i}"

        m_battery += z[i] <= charge_sign[i] * M, f"ZUpper_{i}"
        m_battery += z[i] <= net_buy[i], f"ZNetBuy_{i}"
        m_battery += z[i] >= net_buy[i] - (1 - charge_sign[i]) * M, f"ZLower_{i}"
        m_battery += z[i] >= 0, f"ZNonNeg_{i}"

        m_battery += w[i] <= (1 - charge_sign[i]) * M, f"WUpper_{i}"
        m_battery += w[i] <= net_sell[i], f"WNetSell_{i}"
        m_battery += w[i] >= net_sell[i] - charge_sign[i] * M, f"WLower_{i}"
        m_battery += w[i] >= 0, f"WNonNeg_{i}"

        m_battery += z[i] - w[i] == current_buy_qh[i] + prev_net_trades.loc[i,'net_buy'] - current_sell_qh[i] - prev_net_trades.loc[i,'net_sell'], f"Netting_{i}"


    # set efficiency as sqrt of roundtrip efficiency
    m_battery += lpSum(net_buy[i]*efficiency*1.0/4.0 for i in prices_qh.index) <= max_cycles * cap, "MaxCycles"

    # Solve the problem
    m_battery.solve(GUROBI(msg=0))

    # Solve the problem
    #m_battery.solve(PULP_CBC_CMD(msg=0))

    # print(f"Status: {LpStatus[m_battery.status]}")
    # print(f"Objective value: {m_battery.objective.value()}")

    results = pd.DataFrame(columns=['current_buy_qh', 'current_sell_qh', 'battery_soc'], index=prices_qh.index)

    trades = pd.DataFrame(columns=['execution_time', 'side', 'quantity', 'price', 'product', 'profit'])

    for i in prices_qh.index:
        if (current_buy_qh[i].value() > 0):
            # create buy trade
            new_trade = {'execution_time': [execution_time], 
                        'side': ['buy'], 
                        'quantity': [current_buy_qh[i].value()], 
                        'price': [prices_qh.loc[i,'price']], 
                        'product': [i], 
                        'profit': [-current_buy_qh[i].value() * prices_qh.loc[i,'price'] / 4]}

            # append new trade using concat
            trades = pd.concat([trades, pd.DataFrame(new_trade)], ignore_index=True)

        if (current_sell_qh[i].value() > 0):
            # create sell trade
            new_trade = {'execution_time': [execution_time],
                        'side': ['sell'], 
                        'quantity': [current_sell_qh[i].value()], 
                        'price': [prices_qh.loc[i,'price']], 
                        'product': [i], 
                        'profit': [current_sell_qh[i].value() * prices_qh.loc[i,'price'] / 4]}

            # append new trade using concat
            trades = pd.concat([trades, pd.DataFrame(new_trade)], ignore_index=True)

    for i in prices_qh.index:
        results.loc[i,'current_buy_qh'] = current_buy_qh[i].value()
        results.loc[i,'current_sell_qh'] = current_sell_qh[i].value()
        results.loc[i, 'net_buy'] = net_buy[i].value()
        results.loc[i, 'net_sell'] = net_sell[i].value()
        results.loc[i, 'charge_sign'] = charge_sign[i].value()
        results.loc[i,'battery_soc'] = battery_soc[i].value()

    return results, trades, m_battery.objective.value()

def get_net_trades(trades, end_date):
    # create a new empty dataframe with the columns "net_buy" and "net_sell"
    net_trades = pd.DataFrame(columns=["sum_buy", "sum_sell", "net_buy", "net_sell", 'product'])

    # based on trades, calculate the net buy and net sell for each product
    for product in trades['product'].unique():
        product_trades = trades[trades['product'] == product]
        sum_buy = product_trades[product_trades['side'] == 'buy']['quantity'].sum()
        sum_sell = product_trades[product_trades['side'] == 'sell']['quantity'].sum()
        # add to net_trades using concat
        net_trades = pd.concat([net_trades, pd.DataFrame([[sum_buy, sum_sell, product]], columns=["sum_buy", "sum_sell", 'product'])], ignore_index=True)

    # add the columns "net_buy" and "net_sell" to net_trades, net_buy = sum_buy - sum_sell (if > 0), net_sell = sum_sell - sum_buy (if > 0)
    net_trades['net_buy'] = net_trades['sum_buy'] - net_trades['sum_sell']
    net_trades['net_sell'] = net_trades['sum_sell'] - net_trades['sum_buy']

    # remove values < 0 for net_buy and net_sell
    net_trades.loc[net_trades['net_buy'] < 0, 'net_buy'] = 0
    net_trades.loc[net_trades['net_sell'] < 0, 'net_sell'] = 0

    # set column product to index
    net_trades = net_trades.set_index('product')

    # set start_of_day to end_date minus 1 day
    start_of_day = pd.to_datetime(end_date) - pd.Timedelta(hours=2)

    # set hour and minute to 0 (europe/berlin time)
    start_of_day = start_of_day.replace(hour=0, minute=0)
    end_of_day = start_of_day
    end_of_day = end_of_day.replace(hour=23, minute=45)

    net_trades = net_trades.reindex(pd.date_range(start_of_day, end_of_day, freq='15min'))

    # fill NaN values with 0
    net_trades = net_trades.fillna(0)

    # set index to datetime
    net_trades.index = pd.to_datetime(net_trades.index)

    # return the net_trades dataframe
    return net_trades

def simulate_period(start_day, end_day, threshold, threshold_abs_min, discount_rate, bucket_size, c_rate, roundtrip_eff, max_cycles):

    
    # set path as ./results/threshold
    path = "./results/qh_bs" + str(bucket_size) + 'cr' + str(c_rate) + "rto" + str(roundtrip_eff) + "mc" + str(max_cycles) +  "/"
    tradepath = path + "/trades/"

    # create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(tradepath):
        os.makedirs(tradepath)

    # check if profits.csv exists in path
    if (os.path.exists(path + "/profit.csv")):
        # read profits.csv
        profits = pd.read_csv(path + "/profit.csv")
    else:
        # create profits.csv
        profits = pd.DataFrame(columns=["day", "profit", "cycles"])
    

    if (len(profits) > 0):
        # set current_day to last "day" in profits.csv
        current_day = pd.Timestamp(profits.iloc[-1]["day"], tz='Europe/Berlin') + pd.Timedelta(days=1) + pd.Timedelta(hours=2)
        # set current_cycles to last "cycles" in profits.csv
        current_cycles = profits.iloc[-1]["cycles"]
    else:
        current_day = start_day
        current_cycles = 0

    net_trades = pd.DataFrame(columns=["sum_buy", "sum_sell", "net_buy", "net_sell", 'product'])
    
    while(current_day < end_day):

        current_day = current_day.replace(hour=0, minute=0, second=0, microsecond=0)

        print("current_day: ", current_day)

        all_trades = pd.DataFrame(columns=['execution_time', 'side', 'quantity', 'price', 'product', 'profit'])

        # set trading_start to current_day minus 3 hours
        trading_start = current_day - pd.Timedelta(hours=8)
        # set trading_end to current_day plus 1 day
        trading_end = current_day + pd.Timedelta(days=1)

        print("trading_start: ", trading_start)
        print("trading_end: ", trading_end)

        # set execution_time_start to trading_start
        execution_time_start = trading_start
        # set execution_time_end to trading_start plus 15 minutes
        execution_time_end = trading_start + pd.Timedelta(minutes=bucket_size)

        # calculate number of days until end_day
        days_left = (end_day - current_day).days


        allowed_cycles = max_cycles/365 + ((max_cycles/365 * (365-days_left)) - current_cycles)

        #allowed_cycles = (500 - current_cycles) / days_left
        
        print("Days left: ", days_left)
        print("Current cycles: ", current_cycles)
        print("Allowed cycles: ", allowed_cycles)

        while(execution_time_end < trading_end):
            # get average price for BUY orders
            vwap = get_average_prices("BUY", execution_time_start, execution_time_end, trading_end, min_trades=1)

            #vwap = get_closest_prices(execution_time_start, trading_end)

            net_trades = get_net_trades(all_trades, trading_end)

            # if all vwap["price"] are NaN
            if (vwap["price"].isnull().all()):
                print("No trades in this quarter hour")
                execution_time_start = execution_time_end
                execution_time_end = execution_time_start + pd.Timedelta(minutes=bucket_size)
                continue
            else:
                try:
                    results, trades, profit = run_optimization_quarterhours_repositioning(vwap, execution_time_start, 1, c_rate, roundtrip_eff, allowed_cycles, threshold, threshold_abs_min, discount_rate, net_trades)
                    #append trades to all_trades using concat
                    all_trades = pd.concat([all_trades, trades])
                except:
                    print("Error in optimization")
                    print("execution_time_start: ", execution_time_start)
                    execution_time_start = execution_time_end
                    execution_time_end = execution_time_start + pd.Timedelta(minutes=bucket_size)
                 
                    continue
            
            execution_time_start = execution_time_end
            execution_time_end = execution_time_start + pd.Timedelta(minutes=bucket_size)

        # calculate daily_profit as sum of all_trades["profit"]
        daily_profit = all_trades["profit"].sum()

        current_cycles += net_trades["net_buy"].sum() / 4.0 * roundtrip_eff ** 0.5


        # save trades
        all_trades.to_csv(tradepath + "trades_" + current_day.strftime("%Y-%m-%d") + ".csv", index=False)


        # append daily_profit to profits.csv using concat
        profits = pd.concat([profits, pd.DataFrame([[current_day, daily_profit, current_cycles]], columns=['day', 'profit', 'cycles'])])

        profits_db = pd.DataFrame([[current_day, daily_profit, net_trades["net_buy"].sum() / 4.0 * roundtrip_eff ** 0.5]], columns=['day', 'profit', 'cycles'])

        # add column threshold, threshold_abs and discount_rate to profits_db
        profits_db['type_freq'] = "QH"
        profits_db['max_cycles'] = max_cycles
        profits_db['bucket_size'] = bucket_size
        profits_db['rto'] = roundtrip_eff
        profits_db['c_rate'] = c_rate


        # save profits_db to database
        profits_db.to_sql('revenues_sensitivity', conn_alchemy, if_exists='append', index=False)

        # save profits.csv
        profits.to_csv(path + "profit.csv", index=False)

        # set current day to current_day plus 1 day
        current_day = current_day + pd.Timedelta(days=1) + pd.Timedelta(hours=2)



period_start = pd.Timestamp('2022-01-01 00:00:00', tz='Europe/Berlin')
period_end = pd.Timestamp('2023-01-01 00:00:00', tz='Europe/Berlin')

simulate_period(period_start, period_end, threshold=0, threshold_abs_min=0, discount_rate=0, bucket_size=15, c_rate=0.5, roundtrip_eff=0.86, max_cycles=365)
