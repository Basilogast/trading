import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn-v0_8")

class FindingBacktester(): 
    def optimize_parameters_optuna(self, EMA1_range, EMA2_range, periods_range, rsi_upper_range, rsi_lower_range, signal_mw_range=(5, 21, 1), metric="Multiple", n_trials=100, mode="ema_rsi_combined"):
        '''
        Optimizes EMA_S, EMA_L, RSI periods, RSI upper, RSI lower using Optuna for efficient search.
        '''
        import optuna
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
        else:
            raise ValueError(f"Unknown metric: {metric}")

        def objective(trial):
            if mode == "ema":
                ema_s = trial.suggest_int("EMA_S", EMA1_range[0], EMA1_range[1]-1, step=EMA1_range[2])
                ema_l = trial.suggest_int("EMA_L", EMA2_range[0], EMA2_range[1]-1, step=EMA2_range[2])
                periods = None
                rsi_upper = None
                rsi_lower = None
                signal_mw = None
            elif mode == "rsi":
                ema_s = None
                ema_l = None
                signal_mw = None
                periods = trial.suggest_int("RSI_periods", periods_range[0], periods_range[1]-1, step=periods_range[2])
                rsi_upper = trial.suggest_int("RSI_upper", rsi_upper_range[0], rsi_upper_range[1]-1, step=rsi_upper_range[2])
                rsi_lower = trial.suggest_int("RSI_lower", rsi_lower_range[0], rsi_lower_range[1]-1, step=rsi_lower_range[2])
            elif mode == "macd":
                ema_s = trial.suggest_int("EMA_S", EMA1_range[0], EMA1_range[1]-1, step=EMA1_range[2])
                ema_l = trial.suggest_int("EMA_L", EMA2_range[0], EMA2_range[1]-1, step=EMA2_range[2])
                signal_mw = trial.suggest_int("signal_mw", signal_mw_range[0], signal_mw_range[1]-1, step=signal_mw_range[2])
                periods = None
                rsi_upper = None
                rsi_lower = None
            elif mode == "macd_rsi_combined":
                ema_s = trial.suggest_int("EMA_S", EMA1_range[0], EMA1_range[1]-1, step=EMA1_range[2])
                ema_l = trial.suggest_int("EMA_L", EMA2_range[0], EMA2_range[1]-1, step=EMA2_range[2])
                signal_mw = trial.suggest_int("signal_mw", signal_mw_range[0], signal_mw_range[1]-1, step=signal_mw_range[2])
                periods = trial.suggest_int("RSI_periods", periods_range[0], periods_range[1]-1, step=periods_range[2])
                rsi_upper = trial.suggest_int("RSI_upper", rsi_upper_range[0], rsi_upper_range[1]-1, step=rsi_upper_range[2])
                rsi_lower = trial.suggest_int("RSI_lower", rsi_lower_range[0], rsi_lower_range[1]-1, step=rsi_lower_range[2])
            elif mode == "ema_rsi_combined":
                ema_s = trial.suggest_int("EMA_S", EMA1_range[0], EMA1_range[1]-1, step=EMA1_range[2])
                ema_l = trial.suggest_int("EMA_L", EMA2_range[0], EMA2_range[1]-1, step=EMA2_range[2])
                periods = trial.suggest_int("RSI_periods", periods_range[0], periods_range[1]-1, step=periods_range[2])
                rsi_upper = trial.suggest_int("RSI_upper", rsi_upper_range[0], rsi_upper_range[1]-1, step=rsi_upper_range[2])
                rsi_lower = trial.suggest_int("RSI_lower", rsi_lower_range[0], rsi_lower_range[1]-1, step=rsi_lower_range[2])
                signal_mw = None
            else:
                # Default: suggest all
                ema_s = trial.suggest_int("EMA_S", EMA1_range[0], EMA1_range[1]-1, step=EMA1_range[2])
                ema_l = trial.suggest_int("EMA_L", EMA2_range[0], EMA2_range[1]-1, step=EMA2_range[2])
                periods = trial.suggest_int("RSI_periods", periods_range[0], periods_range[1]-1, step=periods_range[2])
                rsi_upper = trial.suggest_int("RSI_upper", rsi_upper_range[0], rsi_upper_range[1]-1, step=rsi_upper_range[2])
                rsi_lower = trial.suggest_int("RSI_lower", rsi_lower_range[0], rsi_lower_range[1]-1, step=rsi_lower_range[2])
                signal_mw = trial.suggest_int("signal_mw", signal_mw_range[0], signal_mw_range[1]-1, step=signal_mw_range[2])
            if mode == "ema":
                self.prepare_data_EMA_SMA(ema_s, ema_l)
            elif mode == "rsi":
                self.prepare_data_RSI(periods, rsi_upper, rsi_lower)
            elif mode == "macd":
                signal_mw = trial.suggest_int("signal_mw", 5, 20, step=1)
                self.prepare_data_MACD(ema_s, ema_l, signal_mw)
            elif mode == "macd_rsi_combined":
                signal_mw = trial.suggest_int("signal_mw", 5, 20, step=1)
                self.prepare_data_MACD(ema_s, ema_l, signal_mw)
                self.prepare_data_RSI(periods, rsi_upper, rsi_lower)
            elif mode == "ema_rsi_combined":
                self.prepare_data_EMA_SMA(ema_s, ema_l)
                self.prepare_data_RSI(periods, rsi_upper, rsi_lower)
            else:
                # Default: try both
                self.prepare_data_EMA_SMA(ema_s, ema_l)
                self.prepare_data_RSI(periods, rsi_upper, rsi_lower)
            self.run_backtest(mode=mode)
            perf_val = performance_function(self.results["strategy"])
            # Optuna minimizes by default, so return negative performance for maximization
            return -perf_val

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_trial
        print("Optuna optimization completed.")
        if mode == "ema":
            print(f"Best Parameters (EMA/SMA): EMA_S = {best.params['EMA_S']}, EMA_L = {best.params['EMA_L']}")
        elif mode == "rsi":
            print(f"Best Parameters (RSI): RSI_periods = {best.params['RSI_periods']}, RSI_upper = {best.params['RSI_upper']}, RSI_lower = {best.params['RSI_lower']}")
        elif mode == "ema_rsi_combined":
            print(f"Best Parameters (EMA+RSI Combined): EMA_S = {best.params['EMA_S']}, EMA_L = {best.params['EMA_L']}, "
                  f"RSI_periods = {best.params['RSI_periods']}, RSI_upper = {best.params['RSI_upper']}, RSI_lower = {best.params['RSI_lower']}")
        elif mode == "macd":
            print(f"Best Parameters (MACD): EMA_S = {best.params['EMA_S']}, EMA_L = {best.params['EMA_L']}, signal_mw = {best.params.get('signal_mw', 'N/A')}")
        elif mode == "macd_rsi_combined":
            print(f"Best Parameters (MACD + RSI Combination): EMA_S = {best.params['EMA_S']}, EMA_L = {best.params['EMA_L']}, "
                  f"RSI_periods = {best.params['RSI_periods']}, RSI_upper = {best.params['RSI_upper']}, RSI_lower = {best.params['RSI_lower']}, signal_mw = {best.params.get('signal_mw', 'N/A')}")
        else:
            print(f"Best Parameters: EMA_S = {best.params.get('EMA_S', 'N/A')}, EMA_L = {best.params.get('EMA_L', 'N/A')}, "
                  f"RSI_periods = {best.params.get('RSI_periods', 'N/A')}, RSI_upper = {best.params.get('RSI_upper', 'N/A')}, RSI_lower = {best.params.get('RSI_lower', 'N/A')}, signal_mw = {best.params.get('signal_mw', 'N/A')}")
        print(f"Best {self.metric}: {round(-best.value, 6)}")

        # Set the best parameters and run final backtest
        if 'EMA_S' in best.params and 'EMA_L' in best.params:
            self.prepare_data_EMA_SMA(best.params['EMA_S'], best.params['EMA_L'])
        if 'signal_mw' in best.params:
            self.prepare_data_MACD(best.params['EMA_S'], best.params['EMA_L'], best.params['signal_mw'])
        if 'RSI_periods' in best.params and 'RSI_upper' in best.params and 'RSI_lower' in best.params:
            self.prepare_data_RSI(best.params['RSI_periods'], best.params['RSI_upper'], best.params['RSI_lower'])
        self.run_backtest(mode=mode)
        self.print_performance(mode=mode)
    ''' Class for the vectorized backtesting of EMA-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    EMA_S: int
        time window in days for shorter EMA
    EMA_L: int
        time window in days for longer EMA
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    tc: float
        proportional transaction costs per trade
        
        
    Methods
    =======
    get_data:
        retrieves and prepares the data
        
    set_parameters:
        sets one or two new EMA parameters
        
    test_strategy:
        runs the backtest for the EMA-based strategy
        
    plot_results:
        plots the performance of the strategy compared to buy and hold
        
    update_and_run:
        updates EMA parameters and returns the negative absolute performance (for minimization algorithm)
        
    optimize_parameters:
        implements a brute force optimization for the two EMA parameters
    '''
    
    def __init__(self, tc=None, symbol=None):
        '''
        Initialize the FindingBacktester.
        Optionally set transaction cost (tc), symbol, EMA_S, and EMA_L.
        '''
        self.tc = tc
        self.symbol = symbol
    
    def set_parameters(self, SMA = None, EMA = None):
        ''' Updates SMA/EMA parameters and resp. time series.
        '''
        if SMA is not None:
            self.SMA = SMA
            self.data["SMA"] = self.data["price"].rolling(self.SMA).mean() 
        if EMA is not None:
            self.EMA = EMA
            self.data["EMA"] = self.data["price"].ewm(span = self.EMA, min_periods = self.EMA).mean()
        
    def prepare_data_EMA_SMA(self, EMA_S=None, EMA_L=None):
        '''Adds EMA_S and EMA_L columns to self.data. Accepts EMA_S and EMA_L as optional parameters. Sets them as attributes if provided. Also calculates tp_year.'''
        if self.data is not None:
            # Use provided parameters or fall back to instance attributes
            ema_s = EMA_S if EMA_S is not None else getattr(self, 'EMA_S', None)
            ema_l = EMA_L if EMA_L is not None else getattr(self, 'EMA_L', None)
            if ema_s is None or ema_l is None:
                raise ValueError("EMA_S and EMA_L must be provided either as parameters or set as instance attributes.")
            # Set as attributes if provided
            if EMA_S is not None:
                self.EMA_S = EMA_S
            if EMA_L is not None:
                self.EMA_L = EMA_L
            self.data["EMA_S"] = self.data["price"].ewm(span=ema_s, min_periods=ema_s).mean()
            self.data["EMA_L"] = self.data["price"].ewm(span=ema_l, min_periods=ema_l).mean()
            # Calculate tp_year for annualized metrics
            if self.data.index.size > 1:
                days = (self.data.index[-1] - self.data.index[0]).days
                if days > 0:
                    self.tp_year = self.data.shape[0] / (days / 365.25)
                else:
                    self.tp_year = np.nan
            else:
                self.tp_year = np.nan
                
    def prepare_data_RSI(self, periods=None, rsi_upper=None, rsi_lower=None):
        '''Adds RSI columns to self.data. Accepts periods, rsi_upper, rsi_lower as optional parameters. Sets them as attributes if provided.'''
        if self.data is not None:
            periods_val = periods if periods is not None else getattr(self, 'periods', None)
            rsi_upper_val = rsi_upper if rsi_upper is not None else getattr(self, 'rsi_upper', None)
            rsi_lower_val = rsi_lower if rsi_lower is not None else getattr(self, 'rsi_lower', None)
            if periods_val is None or rsi_upper_val is None or rsi_lower_val is None:
                raise ValueError("periods, rsi_upper, and rsi_lower must be provided either as parameters or set as instance attributes.")
            # Set as attributes if provided
            if periods is not None:
                self.periods = periods
            if rsi_upper is not None:
                self.rsi_upper = rsi_upper
            if rsi_lower is not None:
                self.rsi_lower = rsi_lower
            self.data["U"] = np.where(self.data.price.diff() > 0, self.data.price.diff(), 0)
            self.data["D"] = np.where(self.data.price.diff() < 0, -self.data.price.diff(), 0)
            self.data["MA_U"] = self.data.U.rolling(periods_val).mean()
            self.data["MA_D"] = self.data.D.rolling(periods_val).mean()
            self.data["RSI"] = self.data.MA_U / (self.data.MA_U + self.data.MA_D) * 100
            
    def prepare_data_MACD(self, EMA_S=None, EMA_L=None, signal_mw=None):
        '''Adds MACD columns to self.data. Accepts EMA_S, EMA_L, signal_mw as optional parameters. Sets them as attributes if provided.'''
        if self.data is not None:
            ema_s = EMA_S if EMA_S is not None else getattr(self, 'EMA_S', None)
            ema_l = EMA_L if EMA_L is not None else getattr(self, 'EMA_L', None)
            signal_mw_val = signal_mw if signal_mw is not None else getattr(self, 'signal_mw', None)
            if ema_s is None or ema_l is None or signal_mw_val is None:
                raise ValueError("EMA_S, EMA_L, and signal_mw must be provided either as parameters or set as instance attributes.")
            # Set as attributes if provided
            if EMA_S is not None:
                self.EMA_S = EMA_S
            if EMA_L is not None:
                self.EMA_L = EMA_L
            if signal_mw is not None:
                self.signal_mw = signal_mw
            self.data["MACD_EMA_S"] = self.data["price"].ewm(span=ema_s, min_periods=ema_s).mean()
            self.data["MACD_EMA_L"] = self.data["price"].ewm(span=ema_l, min_periods=ema_l).mean()
            self.data["MACD"] = self.data["MACD_EMA_S"] - self.data["MACD_EMA_L"]
            self.data["MACD_Signal"] = self.data["MACD"].ewm(span=signal_mw_val, min_periods=signal_mw_val).mean()
        else:
            ema_s = EMA_S if EMA_S is not None else getattr(self, 'EMA_S', None)
            ema_l = EMA_L if EMA_L is not None else getattr(self, 'EMA_L', None)
            signal_mw_val = signal_mw if signal_mw is not None else getattr(self, 'signal_mw', None)
            if ema_s is None or ema_l is None or signal_mw_val is None:
                raise ValueError("EMA_S, EMA_L, and signal_mw must be provided either as parameters or set as instance attributes.")
            # Set as attributes if provided
            if EMA_S is not None:
                self.EMA_S = EMA_S
            if EMA_L is not None:
                self.EMA_L = EMA_L
            if signal_mw is not None:
                self.signal_mw = signal_mw
            raw = pd.read_csv("forex_pairs.csv", parse_dates=["Date"], index_col="Date")
            raw = raw[self.symbol].to_frame().dropna()
            raw = raw.loc[self.start:self.end]
            raw.rename(columns={self.symbol: "price"}, inplace=True)
            raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))
            raw["MACD_EMA_S"] = raw["price"].ewm(span=ema_s, min_periods=ema_s).mean()
            raw["MACD_EMA_L"] = raw["price"].ewm(span=ema_l, min_periods=ema_l).mean()
            raw["MACD"] = raw["MACD_EMA_S"] - raw["MACD_EMA_L"]
            raw["MACD_Signal"] = raw["MACD"].ewm(span=signal_mw_val, min_periods=signal_mw_val).mean()
            self.data = raw

        # Replace the function definition with the correct signature
        # ...existing code...
            
    def test_strategy(self, mode="ema_rsi_combined"):
        ''' Backtests EMA/SMA, RSI, combined, or EMA/SMA with RSI exit-only filter. '''
        data = self.data.copy().dropna()
        if mode == "ema":
            # EMA/SMA only
            data["position"] = np.where(data["EMA_S"] > data["EMA_L"], 1, -1)
        elif mode == "rsi":
            # RSI only
            rsi_signal = np.where(data["RSI"] > self.rsi_upper, -1, np.nan)
            rsi_signal = np.where(data["RSI"] < self.rsi_lower, 1, rsi_signal)
            data["position"] = pd.Series(rsi_signal, index=data.index).fillna(0)
        elif mode == "ema_rsi_combined":
            # EMA/SMA + RSI Combined
            ema_signal = np.where(data["EMA_S"] > data["EMA_L"], 1, -1)
            rsi_signal = np.where(data["RSI"] > self.rsi_upper, -1, np.nan)
            rsi_signal = np.where(data["RSI"] < self.rsi_lower, 1, rsi_signal)
            rsi_signal = pd.Series(rsi_signal, index=data.index).fillna(0)
            data["position"] = np.where(ema_signal == rsi_signal, ema_signal, 0)
        elif mode == "ema_rsi_exit":
            # EMA/SMA entry, RSI exit-only filter
            data["position"] = np.where(data["EMA_S"] > data["EMA_L"], 1, -1)
            long_exit_mask = (data["position"] == 1) & (data["RSI"] > self.rsi_upper)
            short_exit_mask = (data["position"] == -1) & (data["RSI"] < self.rsi_lower)
            if long_exit_mask.any():
                data.loc[long_exit_mask, "position"] = 0
            if short_exit_mask.any():
                data.loc[short_exit_mask, "position"] = 0
        elif mode == "macd":
            # MACD strategy: MACD line crosses above/below signal line
            data["position"] = np.where(data["MACD"] - data["MACD_Signal"] > 0, 1, -1)
        elif mode == "macd_rsi_combined":
            # MACD and RSI combination strategy
            macd_signal = np.where(data["MACD"] - data["MACD_Signal"] > 0, 1, -1)
            rsi_signal = np.where(data["RSI"] > self.rsi_upper, -1, np.nan)
            rsi_signal = np.where(data["RSI"] < self.rsi_lower, 1, rsi_signal)
            rsi_signal = pd.Series(rsi_signal, index=data.index).fillna(0)
            data["position_MACD"] = macd_signal
            data["position_RSI"] = rsi_signal.astype(int)
            data["position"] = np.where(data["position_MACD"] == data["position_RSI"], data["position_MACD"], 0)
        else:
            raise ValueError("mode must be 'ema', 'rsi', 'ema_rsi_combined', 'ema_rsi_exit', 'macd', or 'macd_rsi_combined'")

        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy - data.trades * self.tc
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        self.print_performance(mode=mode)
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | EMA_S = {} | EMA_L = {} | TC = {}".format(self.symbol, self.EMA_S, self.EMA_L, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
        
    def update_and_run(self, EMA):
        ''' Updates EMA parameters and returns the negative absolute performance (for minimazation algorithm).

        Parameters
        ==========
        EMA: tuple
            EMA parameter tuple
        '''
        self.set_parameters(int(EMA[0]), int(EMA[1]))
        return -self.test_strategy()[0]
    
    def run_backtest(self, mode="ema_rsi_combined"):
        ''' Runs EMA/SMA, RSI, combined, or EMA/SMA with RSI exit-only filter backtest without printing performance. '''
        data = self.data.copy().dropna()
        if mode == "ema":
            data["position"] = np.where(data["EMA_S"] > data["EMA_L"], 1, -1)
        elif mode == "rsi":
            rsi_signal = np.where(data["RSI"] > self.rsi_upper, -1, np.nan)
            rsi_signal = np.where(data["RSI"] < self.rsi_lower, 1, rsi_signal)
            data["position"] = pd.Series(rsi_signal, index=data.index).fillna(0)
        elif mode == "ema_rsi_combined":
            ema_signal = np.where(data["EMA_S"] > data["EMA_L"], 1, -1)
            rsi_signal = np.where(data["RSI"] > self.rsi_upper, -1, np.nan)
            rsi_signal = np.where(data["RSI"] < self.rsi_lower, 1, rsi_signal)
            rsi_signal = pd.Series(rsi_signal, index=data.index).fillna(0)
            data["position"] = np.where(ema_signal == rsi_signal, ema_signal, 0)
        elif mode == "ema_rsi_exit":
            data["position"] = np.where(data["EMA_S"] > data["EMA_L"], 1, -1)
            long_exit_mask = (data["position"] == 1) & (data["RSI"] > self.rsi_upper)
            short_exit_mask = (data["position"] == -1) & (data["RSI"] < self.rsi_lower)
            if long_exit_mask.any():
                data.loc[long_exit_mask, "position"] = 0
            if short_exit_mask.any():
                data.loc[short_exit_mask, "position"] = 0
        elif mode == "macd":
            data["position"] = np.where(data["MACD"] - data["MACD_Signal"] > 0, 1, -1)
        elif mode == "macd_rsi_combined":
            macd_signal = np.where(data["MACD"] - data["MACD_Signal"] > 0, 1, -1)
            rsi_signal = np.where(data["RSI"] > self.rsi_upper, -1, np.nan)
            rsi_signal = np.where(data["RSI"] < self.rsi_lower, 1, rsi_signal)
            rsi_signal = pd.Series(rsi_signal, index=data.index).fillna(0)
            data["position_MACD"] = macd_signal
            data["position_RSI"] = rsi_signal.astype(int)
            data["position"] = np.where(data["position_MACD"] == data["position_RSI"], data["position_MACD"], 0)
        else:
            raise ValueError("mode must be 'ema', 'rsi', 'ema_rsi_combined', 'ema_rsi_exit', 'macd', or 'macd_rsi_combined'")

        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy - data.trades * self.tc
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

    def find_best_strategy(self, mode="ema_rsi_combined"):
        ''' Finds the optimal strategy (global maximum) given the parameter ranges. '''
        best = self.results_overview.nlargest(1, "Performance")
        EMA_S = int(best.EMA_S.iloc[0])
        EMA_L = int(best.EMA_L.iloc[0])
        perf = best.Performance.iloc[0]
        # Print combined strategy parameters including RSI if available
        rsi_info = ""
        if hasattr(self, "periods") and hasattr(self, "rsi_upper") and hasattr(self, "rsi_lower"):
            rsi_info = f", RSI Periods = {self.periods}, RSI Upper = {self.rsi_upper}, RSI Lower = {self.rsi_lower}"
        print(f"Best Parameters: EMA_S = {EMA_S}, EMA_L = {EMA_L}{rsi_info}, {self.metric}: {round(perf, 6)}")
        self.prepare_data_EMA_SMA(EMA_S, EMA_L)
        # Use EMA+RSI combined logic for backtest and reporting
        self.run_backtest(mode=mode)
        self.print_performance(mode=mode)
    
    def download_recent_data(self, instrument, count=500, file_name="recent_data.csv", granularity="M1"):
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
                    "granularity": granularity,  # Use user-specified time frame
                    "count": rows_to_fetch,
                    "price": "MBA"  # Request mid, bid, and ask prices for spread calculation
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
                        open_price = float(candle["mid"]["o"])
                        high_price = float(candle["mid"]["h"])
                        low_price = float(candle["mid"]["l"])
                        close_price = float(candle["mid"]["c"])
                        price = close_price
                        spread = float(candle.get("ask", {}).get("c", 0)) - float(candle.get("bid", {}).get("c", 0))  # Spread
                        batch_data.append({
                            "time": time,
                            "price": price,
                            "Open": open_price,
                            "High": high_price,
                            "Low": low_price,
                            "Close": close_price,
                            "spread": spread
                        })
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
    
    def print_performance(self, leverage=False, mode="ema_rsi_combined"):
        ''' Calculates and prints various Performance Metrics, with strategy name based on mode. '''
        data = self.results.copy()
        if leverage:
            to_analyze = np.log(data.strategy_levered.add(1))
        else:
            to_analyze = data.strategy

        strategy_multiple = round(self.calculate_multiple(to_analyze), 6)
        bh_multiple = round(self.calculate_multiple(data.returns), 6)
        outperf = round(strategy_multiple - bh_multiple, 6)
        cagr = round(self.calculate_cagr(to_analyze), 6)
        ann_mean = round(self.calculate_annualized_mean(to_analyze), 6)
        ann_std = round(self.calculate_annualized_std(to_analyze), 6)
        sharpe = round(self.calculate_sharpe(to_analyze), 6)
        sortino = round(self.calculate_sortino(to_analyze), 6)
        max_drawdown = round(self.calculate_max_drawdown(to_analyze), 6)
        calmar = round(self.calculate_calmar(to_analyze), 6)
        max_dd_duration = round(self.calculate_max_dd_duration(to_analyze), 6)
        kelly_criterion = round(self.calculate_kelly_criterion(to_analyze), 6)

        print(100 * "=")
        if mode == "ema":
            strategy_name = "EMA/SMA STRATEGY"
        elif mode == "rsi":
            strategy_name = "RSI STRATEGY"
        elif mode == "ema_rsi_combined":
            strategy_name = "EMA/SMA + RSI COMBINED STRATEGY"
        elif mode == "macd":
            strategy_name = "MACD STRATEGY"
        elif mode == "macd_rsi_combined":
            strategy_name = "MACD + RSI COMBINATION STRATEGY"
        else:
            strategy_name = "CUSTOM STRATEGY"
        print(f"{strategy_name} | INSTRUMENT = {self.symbol}")
        print(100 * "-")
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
