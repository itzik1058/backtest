from abc import abstractmethod
from typing import Optional, Tuple, Callable
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class PortfolioStrategy:
    def __call__(self, price_history: pd.DataFrame, wealth: float) -> pd.Series:
        portfolio = self.get_portfolio(price_history, wealth)
        assert portfolio.ge(0).all(), 'Shorts are not allowed'
        return portfolio
    
    @abstractmethod
    def get_portfolio(self, price_history: pd.DataFrame, wealth: float) -> pd.Series:
        raise NotImplementedError


class PortfolioAllocationStrategy(PortfolioStrategy):
    def get_portfolio(self, price_history: pd.DataFrame, wealth: float) -> pd.Series:
        allocation = self.get_allocation(price_history, wealth)
        assert allocation.ge(0).all(), 'Shorts are not allowed'
        allocation = allocation * wealth / allocation.sum()
        prices = price_history.iloc[-1]
        prices = prices.fillna(0)
        prices['$'] = 1
        portfolio = (allocation // prices).fillna(0)
        portfolio['$'] += (allocation - portfolio * prices).sum()
        return portfolio

    @abstractmethod
    def get_allocation(self, price_history: pd.DataFrame, wealth: float) -> pd.Series:
        raise NotImplementedError


class Backtest:
    def __init__(self, prices: pd.DataFrame, strategy: PortfolioStrategy, initial_wealth: float):
        assert '$' not in prices.columns
        self.prices = prices.dropna(axis=0, how='all').copy()
        self.prices['$'] = 1
        self.strategy = strategy
        self.initial_wealth = initial_wealth
        self.portfolio = None
    
    def _portfolio_worth(self, prices: pd.Series):
        return self.portfolio @ prices.fillna(0)

    def _returns_for_date(self, date: np.datetime64) -> Tuple[pd.Series, float]:
        prices = self.prices[:date]
        na = prices.iloc[-1].isna()
        available = prices.columns[~na]
        if prices.shape[0] >= 2:
            self.portfolio[prices.columns[na]] = 0
            self.portfolio[available] = self.strategy(prices[available].iloc[:-1],
                                           self._portfolio_worth(prices.iloc[-2]))
        return self._portfolio_worth(prices.iloc[-1])
    
    def run(self,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            min_wealth: float = 0, weekday: Optional[int] = 0,
            verbose: bool = False, callback: Optional[Callable] = None):
        self.portfolio = pd.Series(np.zeros(self.prices.shape[1]), index=self.prices.columns)
        self.portfolio['$'] = self.initial_wealth
        wealth = self.initial_wealth
        returns = []
        dates = self.prices[start_date:end_date].index
        if weekday is not None:
            dates = dates[dates.weekday == weekday]
        dates_iter = dates if not verbose or tqdm is None else tqdm(dates)
        for date in dates_iter:
            new_wealth = self._returns_for_date(date)
            ret = new_wealth / wealth - 1
            wealth = new_wealth
            returns.append(ret)
            if verbose or callback is not None:
                rmin, rmax = np.min(returns), np.max(returns)
                rmean, rstd = np.mean(returns), np.std(returns)
                rgmean = scipy.stats.gmean(np.array(returns) + 1) - 1
            if callback is not None:
                callback({
                    'Date': date,
                    'Return': ret,
                    'Wealth': wealth,
                    'Min Return': rmin,
                    'Max Return': rmax,
                    'Return Geometric Mean': rgmean,
                    'Return Mean': rmean,
                    'Return Std': rstd,
                    'Portfolio': self.portfolio[self.portfolio.gt(0)].to_dict()
                })
            if verbose and tqdm is not None:
                dates_iter.set_postfix(date=date.strftime('%Y-%m-%d'), wealth=f'{wealth:.2f}$',
                                       rmin=f'{rmin:.2%}', rgmean=f'{rgmean:.2%}', rstd=f'{rstd:.2%}',
                                       portfolio=' '.join({f'{k}:{int(v)}' for k, v in self.portfolio.to_dict().items() if v != 0}))
            if wealth < min_wealth:
                break
        return pd.Series(returns, index=dates[:len(returns)], name='Return')
