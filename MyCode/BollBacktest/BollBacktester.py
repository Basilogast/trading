import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
import os
import logging

plt.style.use("seaborn-v0_8")

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class BollBacktester():
    ''' Class for the vectorized backtesting of Bollinger Bands trading strategies.
    
    Attributes
    ============
    filepath: str
        local filepath of the dataset (csv-file)
    symbol: str
        ticker symbol (instrument) to be backtested
    start: str
        start date for data import
    end: str
        end date for data import
    tc: float
        proportional trading costs per trade
    
    
    Methods
    =======
    get_data:
        imports the data.
        
    test_strategy:
        prepares the data and backtests the trading strategy incl. reporting (wrapper).
        
    prepare_data:
        prepares the data for backtesting.
    
    run_backtest:
        runs the strategy backtest.
        
    upsample:
        upsamples/copies trading positions back to higher frequency.
        
    plot_results:
        plots the cumulative performance of the trading strategy compared to buy-and-hold.
        
    optimize_strategy:
        backtests strategy for different parameter values incl. optimization and reporting (wrapper).
    
    find_best_strategy:
        finds the optimal strategy (global maximum) given the parameter ranges.
        
    visualize_many:
        plots parameter values vs. performance. 
        
    add_sessions:
        adds/labels trading sessions and their compound returns.
    
    add_stop_loss:
        adds stop loss to the strategy.
    
    add_take_profit: 
        adds take profit to the strategy.
        
    add_leverage:
        adds leverage to the strategy.
        
    print_performance:
        calculates and prints various performance metrics.
        
    '''    
    
    def __init__(self, filepath, symbol, start, end, tc):
        
        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.get_data()
        self.tp_year = (self.data.price.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))
        
    def __repr__(self):
        return "BollBacktester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)
        
    def get_data(self):
        ''' Imports the data.
        '''
        raw = pd.read_csv(self.filepath, parse_dates = ["time"], index_col = "time")
        raw = raw[self.symbol].to_frame().ffill()
        raw = raw.loc[self.start:self.end].copy()
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw.price / raw.price.shift(1))
        self.data = raw
        
    def get_data_oanda(self, filepath, start_date, end_date):
        '''
        Imports the data based on a date range and a specified file path.

        Parameters
        ==========
        filepath: str
            Path to the CSV file containing the data.
        start_date: str
            Start date for the data (format: 'YYYY-MM-DD').
        end_date: str
            End date for the data (format: 'YYYY-MM-DD').
        '''
        # Read the CSV file
        raw = pd.read_csv(filepath, parse_dates=["time"], index_col="time")

        # Filter data based on the date range
        raw = raw.loc[start_date:end_date].copy()

        # Rename the price column and calculate returns
        raw.rename(columns={"price": "price"}, inplace=True)
        raw["returns"] = np.log(raw.price / raw.price.shift(1))

        # Save the filtered data to self.data
        self.data = raw

    def test_strategy(self, freq = 60, window = 50, dev = 2): # Adj!!!
        '''
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).
         
        Parameters
        ============
        freq: int
            data frequency/granularity to work with (in minutes)
        
        window: int
            time window (number of bars) to calculate the simple moving average price (SMA).
            
        dev: int
            number of standard deviations to calculate upper and lower bands.
        '''
        self.freq = "{}min".format(freq) 
        self.window = window
        self.dev = dev # NEW!!!
                                
        self.prepare_data(freq, window, dev) 
        self.upsample() 
        self.run_backtest()
        
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        self.print_performance()
    
    def prepare_data(self, freq, window, dev): # Adj!!!
        ''' Prepares the Data for Backtesting.
        '''
        # Ensure the time column is in datetime format and set as the index

        data = self.data.price.to_frame().copy()
        freq = "{}min".format(freq)
        resamp = data.resample(freq).last().dropna().iloc[:-1]
        
        ######### INSERT THE STRATEGY SPECIFIC CODE HERE ##################
        resamp["SMA"] = resamp["price"].rolling(window).mean()
        resamp["Lower"] = resamp["SMA"] - resamp["price"].rolling(window).std() * dev
        resamp["Upper"] = resamp["SMA"] + resamp["price"].rolling(window).std() * dev
        
        resamp["distance"] = resamp.price - resamp.SMA
        resamp["position"] = np.where(resamp.price < resamp.Lower, 1, np.nan)
        resamp["position"] = np.where(resamp.price > resamp.Upper, -1, resamp["position"])
        resamp["position"] = np.where(resamp.distance * resamp.distance.shift(1) < 0, 0, resamp["position"])
        resamp["position"] = resamp.position.ffill().fillna(0)
        ###################################################################

        resamp.dropna(inplace = True)
        self.results = resamp
        return resamp 
    

    def export_resampled_data(self, freq, window, dev, file_name):
        """
        Resamples the data using the prepare_data method and exports it to a specified CSV file.

        Parameters
        ==========
        freq: int
            Data frequency/granularity to work with (in minutes).
        window: int
            Time window (number of bars) to calculate the simple moving average price (SMA).
        dev: int
            Number of standard deviations to calculate upper and lower bands.
        file_name: str
            The name of the CSV file to be saved.
        """
        try:
            # Call the prepare_data method to resample the data
            self.prepare_data(freq, window, dev)
            
            # Construct the file path dynamically
            parent_folder = os.path.dirname(os.path.dirname(__file__))
            data_folder = os.path.join(parent_folder, 'data')
            os.makedirs(data_folder, exist_ok=True)  # Ensure the 'data' folder exists
            file_path = os.path.join(data_folder, file_name)
            
            # Export the resampled data stored in self.results to the specified file path
            self.results.to_csv(file_path)
            print(f"Resampled data successfully exported to {file_path}")
        except Exception as e:
            # logging.error(f"An unexpected error occurred: {e}")
            raise

        # logging.debug("Finished prepare_data method.")

    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''
        
        data = self.results.copy()

        # Ensure the 'returns' column exists
        if "returns" not in data.columns:
            data["returns"] = np.log(data["price"] / data["price"].shift(1))

        data["strategy"] = data["position"].shift(1) * data["returns"]
        
        # determine the number of trades in each bar
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # subtract transaction/trading costs from pre-cost return
        data.strategy = data.strategy - data.trades * self.tc
        
        self.results = data
        
    def upsample(self):
        ''' Upsamples/copies trading positions back to higher frequency. '''
        # logging.debug("Entering upsample function.")
        # logging.debug("State of self.results before creating resamp:")
        # logging.debug(self.results)

        if self.results.empty:
            # logging.warning("self.results is empty. Cannot create resamp.")
            return

        resamp = self.results.copy()
        # logging.debug("State of resamp after copying self.results:")
        # logging.debug(resamp)

        if resamp.empty:
            # logging.warning("resamp is empty after copying self.results.")
            return

        # Ensure resamp has data before accessing index[0]
        if resamp.index.size == 0:
            # logging.warning("resamp index is empty. Cannot proceed with upsample.")
            return

        try:
            # logging.debug("State of self.data before slicing with resamp index:")
            # logging.debug(self.data)

            # Ensure alignment between resamp and self.data indices
            aligned_index = self.data.index.union(resamp.index)
            self.data = self.data.reindex(aligned_index).ffill()  # Forward fill missing values

            # Slice self.data using resamp index
            data = self.data.loc[resamp.index[0]:].copy()
            data["position"] = resamp.position.shift()
            data.position = data.position.shift(-1).ffill()

            self.results = data
            # logging.debug("State of self.results after upsample:")
            # logging.debug(self.results)

        except KeyError as e:
            # logging.error(f"KeyError during upsample: {e}")
            raise

        except Exception as e:
            # logging.error(f"An error occurred during upsample: {e}")
            raise
    
    def plot_results(self, leverage = False):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        elif leverage:
            title = "{} | Window = {} | Frequency = {} | TC = {} | Leverage = {}".format(self.symbol, self.window, self.freq, self.tc, self.leverage)
            self.results[["creturns", "cstrategy", "cstrategy_levered"]].plot(title=title, figsize=(12, 8))
        else:
            title = "{} | Window = {} | Frequency = {} | TC = {}".format(self.symbol, self.window, self.freq, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            
    def optimize_strategy(self, freq_range, window_range, dev_range, metric = "Multiple"): # Adj!!!
        '''
        Backtests strategy for different parameter values incl. Optimization and Reporting (Wrapper).
         
        Parameters
        ============
        freq_range: tuple
            tuples of the form (start, end, step size).
        
        window_range: tuple
            tuples of the form (start, end, step size).
        
        dev_range: tuple
            tuples of the form (start, end, step size).
        
        metric: str
            performance metric to be optimized (can be: "Multiple", "Sharpe", "Sortino", "Calmar", "Kelly")
        '''
        
        # logging.debug("Starting optimize_strategy method.")
        # logging.debug(f"Parameters received: freq_range={freq_range}, window_range={window_range}, dev_range={dev_range}, metric={metric}")

        self.metric = metric
        
        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe
        elif metric == "Sortino":
            performance_function = self.calculate_sortino
        elif metric == "Calmar": 
            performance_function = self.calculate_calmar
        elif metric == "Kelly": 
            performance_function = self.calculate_kelly_criterion
        
        freqs = range(*freq_range)  
        windows = range(*window_range)
        devs = np.arange(*dev_range) # NEW!!!

        # Debugging: Print the devs range
        print("Devs range:", list(devs))

        combinations = list(product(freqs, windows, devs)) 

        # Debugging: Print the first few combinations
        print("First 5 combinations:", combinations[:5])

        performance = []
        for comb in combinations: 
            self.prepare_data(comb[0], comb[1], comb[2])
            # logging.debug(f"Testing combination index : freq={comb[0]}, window={comb[1]}, dev={comb[2]}")
            # logging.debug("State of self.data after prepare_data:")
            # logging.debug(self.data)
            self.upsample()
            # logging.debug("State of self.results after upsample:")
            # logging.debug(self.results)
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))
    
        self.results_overview = pd.DataFrame(
            data=np.array(combinations),
            columns=["Freq", "Windows", "Devs"]  # Ensure "Devs" is included here
        )

        # Debugging: Print the results_overview DataFrame
        print("Results overview head:")
        print(self.results_overview.head())

        self.results_overview["Performance"] = performance
        self.find_best_strategy()
        
    def find_best_strategy(self):
        ''' Finds the optimal strategy (global maximum) given the parameter ranges.
        '''
        best = self.results_overview.nlargest(1, "Performance")
        freq = int(best.Freq.iloc[0]) 
        window = int(best.Windows.iloc[0])
        dev = best.Devs.iloc[0] # NEW!!!
        perf = best.Performance.iloc[0]
        print("Frequency: {} | Windows: {} | Devs: {} | {}: {}".format(freq, window, dev, self.metric, round(perf, 6))) 
        self.test_strategy(freq, window, dev) 
        
    def visualize_many(self):
        ''' Plots parameter values vs. Performance, ensuring uniqueness of Freq, Windows, and Devs.
        '''

        if self.results_overview is None:
            print("Run optimize_strategy() first.")
        else: 
            # Ensure 'Devs' is included in the aggregation
            self.results_overview = self.results_overview.groupby(["Freq", "Windows", "Devs"], as_index=False).agg({"Performance": "mean"})

            # Reset index to ensure no duplicate index issues
            self.results_overview.reset_index(drop=True, inplace=True)

            # Diagnostic: Print DataFrame info and first few rows
            # print("DataFrame Info:")
            # print(self.results_overview.info())
            # print("First few rows of the DataFrame:")
            # print(self.results_overview.head())

            # Pivot the data for heatmap, considering Freq, Windows, and Devs for uniqueness
            matrix = self.results_overview.pivot_table(index="Freq", columns=["Windows", "Devs"], values="Performance")
            # print("Pivot operation successful. Matrix shape:", matrix.shape)

            plt.figure(figsize=(12, 8))
            sns.set_theme(font_scale=1.5)
            sns.heatmap(matrix, cmap="RdYlGn", robust=True, cbar_kws={"label": "{}".format(self.metric)})
            plt.title("Performance Heatmap (Unique Freq, Windows, and Devs)")
            plt.show()
            
    def add_sessions(self, visualize = False):
        ''' 
        Adds/Labels Trading Sessions and their compound returns.
        
        Parameter
        ============
        visualize: bool, default False
            if True, visualize compound session returns over time
        '''
        
        if self.results is None:
            print("Run test_strategy() first.")
            
        data = self.results.copy()
        data["session"] = np.sign(data.trades).cumsum().shift().fillna(0)
        data["session_compound"] = data.groupby("session").strategy.cumsum().apply(np.exp) - 1
        self.results = data
        if visualize:
            data["session_compound"].plot(figsize = (12, 8))
            plt.show()
    
    def add_stop_loss(self, sl_thresh, report = True): 
        ''' 
        Adds Stop Loss to the Strategy.
        
        Parameter
        ============
        sl_thresh: float (negative)
            maximum loss level in % (e.g. -0.02 for -2%)
        
        report: bool, default True
            if True, print Performance Report incl. Stop Loss. 
        '''
        
        self.sl_thresh = sl_thresh
        
        if self.results is None:
            print("Run test_strategy() first.")
        
        self.add_sessions()
        self.results = self.results.groupby("session", group_keys = False).apply(self.define_sl_pos, include_groups = False)
        self.run_backtest()
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        self.add_sessions()
        
        if report:
            self.print_performance()
            
            
    def add_take_profit(self, tp_thresh, report = True):
        ''' 
        Adds Take Profit to the Strategy.
        
        Parameter
        ============
        tp_thresh: float (positive)
            maximum profit level in % (e.g. 0.02 for 2%)
        
        report: bool, default True
            if True, print Performance Report incl. Take Profit. 
        '''
        self.tp_thresh = tp_thresh
        
        if self.results is None:
            print("Run test_strategy() first.")
        
        self.add_sessions()
        self.results = self.results.groupby("session", group_keys = False).apply(self.define_tp_pos, include_groups = False)
        self.run_backtest()
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        self.add_sessions()
        
        if report:
            self.print_performance()
        
    def define_sl_pos(self, group):
        if (group.session_compound <= self.sl_thresh).any():
            start = group[group.session_compound <= self.sl_thresh].index[0]
            stop = group.index[-2]
            group.loc[start:stop, "position"] = 0
            return group
        else:
            return group 
        
    def define_tp_pos(self, group):
        if (group.session_compound >= self.tp_thresh).any():
            start = group[group.session_compound >= self.tp_thresh].index[0]
            stop = group.index[-2]
            group.loc[start:stop, "position"] = 0
            return group
        else:
            return group
        
    def add_leverage(self, leverage, sl = -0.5, report = True):
        ''' 
        Adds Leverage to the Strategy.
        
        Parameter
        ============
        leverage: float (positive)
            degree of leverage.
        
        sl: float (negative), default -50% (regulatory)
            maximum margin loss level in % (e.g. -0.2 for -20%).
        
        report: bool, default True
            if True, print Performance Report incl. Leverage.
        '''
        
        self.leverage = leverage
        sl_thresh = sl / leverage
        self.add_stop_loss(sl_thresh, report = False)
        
        data = self.results.copy()
        data["simple_ret"] = np.exp(data.strategy) - 1
        data["eff_lev"] = leverage * (1 + data.session_compound) / (1 + data.session_compound * leverage)
        data["eff_lev"] = data.eff_lev.fillna(leverage)
        data.loc[data.trades !=0, "eff_lev"] = leverage
        levered_returns = data.eff_lev.shift() * data.simple_ret
        levered_returns = np.where(levered_returns < -1, -1, levered_returns)
        data["strategy_levered"] = levered_returns
        data["cstrategy_levered"] = data.strategy_levered.add(1).cumprod()
        
        self.results = data
            
        if report:
            self.print_performance(leverage = True)
    ############################## Performance ######################################
    
    def print_performance(self, leverage = False):
        ''' Calculates and prints various Performance Metrics.
        '''
        
        data = self.results.copy()
        
        if leverage:
            to_analyze = np.log(data.strategy_levered.add(1))
        else: 
            to_analyze = data.strategy
        
        strategy_multiple = round(self.calculate_multiple(to_analyze), 6)
        bh_multiple =       round(self.calculate_multiple(data.returns), 6)
        outperf =           round(strategy_multiple - bh_multiple, 6)
        cagr =              round(self.calculate_cagr(to_analyze), 6)
        ann_mean =          round(self.calculate_annualized_mean(to_analyze), 6)
        ann_std =           round(self.calculate_annualized_std(to_analyze), 6)
        sharpe =            round(self.calculate_sharpe(to_analyze), 6)
        sortino =           round(self.calculate_sortino(to_analyze), 6)
        max_drawdown =      round(self.calculate_max_drawdown(to_analyze), 6)
        calmar =            round(self.calculate_calmar(to_analyze), 6)
        max_dd_duration =   round(self.calculate_max_dd_duration(to_analyze), 6)
        kelly_criterion =   round(self.calculate_kelly_criterion(to_analyze), 6)
        
        print(100 * "=")
        print("SIMPLE BOLLINGER BAND STRATEGY | INSTRUMENT = {} | Freq: {} | WINDOW = {}".format(self.symbol, self.freq, self.window))
        print(100 * "-")
        #print("\n")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Multiple (Strategy):         {}".format(strategy_multiple))
        print("Multiple (Buy-and-Hold):     {}".format(bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {}".format(outperf))
        print("\n")
        print("CAGR:                        {}".format(cagr))
        print("Annualized Mean:             {}".format(ann_mean))
        print("Annualized Std:              {}".format(ann_std))
        print("Sharpe Ratio:                {}".format(sharpe))
        print("Sortino Ratio:               {}".format(sortino))
        print("Maximum Drawdown:            {}".format(max_drawdown))
        print("Calmar Ratio:                {}".format(calmar))
        print("Max Drawdown Duration:       {} Days".format(max_dd_duration))
        print("Kelly Criterion:             {}".format(kelly_criterion))
        
        print(100 * "=")
        
    def calculate_multiple(self, series):
        return np.exp(series.sum())
    
    def calculate_cagr(self, series):
        return np.exp(series.sum())**(1/((series.index[-1] - series.index[0]).days / 365.25)) - 1
    
    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year
    
    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)
    
    def calculate_sharpe(self, series):
        if series.std() == 0:
            return np.nan
        else:
            return series.mean() / series.std() * np.sqrt(self.tp_year)
    
    def calculate_sortino(self, series):
        excess_returns = (series - 0)
        downside_deviation = np.sqrt(np.mean(np.where(excess_returns < 0, excess_returns, 0)**2))
        if downside_deviation == 0:
            return np.nan
        else:
            sortino = (series.mean() - 0) / downside_deviation * np.sqrt(self.tp_year)
            return sortino 
    
    def calculate_max_drawdown(self, series):
        creturns = series.cumsum().apply(np.exp)
        cummax = creturns.cummax()
        drawdown = (cummax - creturns)/cummax
        max_dd = drawdown.max()
        return max_dd
    
    def calculate_calmar(self, series):
        max_dd = self.calculate_max_drawdown(series)
        if max_dd == 0:
            return np.nan
        else:
            cagr = self.calculate_cagr(series)
            calmar = cagr / max_dd
            return calmar
    
    def calculate_max_dd_duration(self, series):
        creturns = series.cumsum().apply(np.exp)
        cummax = creturns.cummax()
        drawdown = (cummax - creturns)/cummax
    
        begin = drawdown[drawdown == 0].index
        end = begin[1:]
        end = end.append(pd.DatetimeIndex([drawdown.index[-1]]))
        periods = end - begin
        max_ddd = periods.max()
        return max_ddd.days
    
    def calculate_kelly_criterion(self, series):
        series = np.exp(series) - 1
        if series.var() == 0:
            return np.nan
        else:
            return series.mean() / series.var()
    
    ############################## Performance ######################################
    def download_recent_data(self, instrument, count=500, file_name="recent_data.csv"):
        """
        Downloads recent data from OANDA in the one-minute time frame, including time, price, and spread,
        saves it in `self.data`, and exports it to a CSV file in the `data` folder on the same level as this file.

        Parameters
        ==========
        instrument: str
            The trading instrument (e.g., 'EUR_USD').
        count: int, default 500
            The number of recent data points to fetch.
        file_name: str, default "recent_data.csv"
            The name of the CSV file to save the data.
        """
        import oandapyV20
        import oandapyV20.endpoints.instruments as instruments
        import os
        import configparser

        try:
            # Read the access token from the oanda.cfg file
            config = configparser.ConfigParser()
            config.read(os.path.join(os.path.dirname(__file__), 'oanda.cfg'))
            api_key = config['oanda']['access_token']

            # Initialize the OANDA API client
            client = oandapyV20.API(access_token=api_key)

            # Define the parameters for fetching data
            max_rows_per_request = 5000  # Adjust based on API limits
            data = []
            last_time = None
            download_count = 0  # Track the number of downloads

            # Construct the file path dynamically
            data_folder = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(data_folder, exist_ok=True)  # Ensure the 'data' folder exists
            file_path = os.path.join(data_folder, file_name)

            # Fetch data in batches
            while count > 0:
                rows_to_fetch = min(count, max_rows_per_request)
                params = {
                    "granularity": "M1",  # One-minute time frame
                    "count": rows_to_fetch
                }
                if last_time:
                    params["to"] = last_time  # Adjust to fetch the next batch of rows

                # logging.info(f"Performing download #{download_count + 1} with {rows_to_fetch} rows.")
                # logging.debug(f"Request parameters: {params}")
                r = instruments.InstrumentsCandles(instrument=instrument, params=params)
                client.request(r)
                download_count += 1

                # Parse the response
                candles = r.response.get("candles", [])
                if not candles:
                    # logging.warning("No candle data received from API.")
                    break

                batch_data = []
                for candle in candles:
                    try:
                        time = pd.to_datetime(candle["time"]).strftime('%Y-%m-%d %H:%M:%S')  # Reformat time
                        price = (float(candle["mid"]["o"])+float(candle["mid"]["c"]))/2  # Average of open and close
                        spread = float(candle.get("ask", {}).get("c", 0)) - float(candle.get("bid", {}).get("c", 0))  # Spread
                        batch_data.append({"time": time, "price": price, "spread": spread})
                    except KeyError as e:
                        # logging.warning(f"Missing field in candle data: {e}")
                        pass

                # Update last_time to the timestamp of the first candle in the batch
                if candles:
                    last_time = pd.to_datetime(candles[0]["time"]).strftime('%Y-%m-%dT%H:%M:%SZ')
                    # logging.debug(f"Updated last_time to: {last_time}")
                else:
                    # logging.warning("No new data received. Breaking the loop.")
                    break

                count -= rows_to_fetch

                # Convert batch data to a DataFrame and append to the file
                batch_df = pd.DataFrame(batch_data)
                batch_df["time"] = pd.to_datetime(batch_df["time"])
                batch_df.set_index("time", inplace=True)  # Ensure DatetimeIndex
                batch_df["returns"] = np.log(batch_df["price"] / batch_df["price"].shift(1))  # Add returns column

                # Deduplicate rows before appending
                if os.path.exists(file_path):
                    existing_data = pd.read_csv(file_path, parse_dates=["time"], index_col="time")
                    batch_df = batch_df[~batch_df.index.isin(existing_data.index)]

                # Append to the CSV file
                if os.path.exists(file_path):
                    batch_df.to_csv(file_path, mode='a', header=False)
                else:
                    batch_df.to_csv(file_path)

                # logging.info(f"Batch of {len(batch_df)} rows successfully appended to {file_path}")

                # Append batch data to self.data
                data.extend(batch_data)

            # Log the total number of downloads
            # logging.info(f"Total downloads performed: {download_count}")

            # Log the state of the data
            self.data = pd.DataFrame(data)
            self.data["time"] = pd.to_datetime(self.data["time"])  # Ensure DatetimeIndex for self.data
            self.data.set_index("time", inplace=True)
            self.data["returns"] = np.log(self.data["price"] / self.data["price"].shift(1))  # Add returns column to self.data
            # logging.debug("State of self.data after processing:")
            # logging.debug(self.data.head())

            # Sort the final data in chronological order
            self.data.sort_index(inplace=True)

            # Save the sorted data back to the CSV file
            self.data.to_csv(file_path)
        except Exception as e:
            # logging.error(f"An error occurred while downloading data: {e}")
            raise

        # logging.debug("Finished download_recent_data method.")