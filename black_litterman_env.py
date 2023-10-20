import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pypfopt import (
    BlackLittermanModel,
    EfficientFrontier,
    black_litterman,
    objective_functions,
    plotting,
    risk_models,
)

import warnings
warnings.simplefilter("ignore", UserWarning)

class BlackLittermanEnv(gym.Env):
    def __init__(
        self,
        data_df: pd.DataFrame,
        information_cols: list = ["Open", "High", "Low", "Adj Close", "Volume"],
        other_df_path:str = 'Other.csv',
        other_cols: list = ['10_year_rate','3_months_rate'],
        if_confidence: bool = True,
        stock1_weight: float = 0.4,
        stock2_weight: float = 0.6,
        initial_amount: float = 1000_000,
        transaction_cost_pct: float = 0.001,
        base_return: float = 0.02,
        base_return_penalty: float = 2000,
    ):

        self.data_df = data_df

        self.stock_dim = 2
        self.state_space_shape = self.stock_dim
        self.if_confidence = if_confidence
        self.information_cols = information_cols
        self.other_cols = other_cols
        self.transaction_cost_pct = transaction_cost_pct
        self.base_return = base_return
        self.base_return_penalty = -base_return_penalty
        self.actions = []
        self.return_action_space = spaces.Box(
            low=-1, high=+1, shape=(self.stock_dim + 1,), dtype=float
        )

        if self.if_confidence:
            self.confidence_action_space = spaces.Box(
                low=0, high=+1, shape=(self.stock_dim + 1,), dtype=float
            )
            self.action_space = spaces.Tuple(
                (self.return_action_space, self.confidence_action_space)
            )
        else:
            self.action_space = self.return_action_space

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.information_cols)+1, self.state_space_shape),
        )

        self.terminal = False
        self.initial_amount = initial_amount
        self.portfolio_value = initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [[0] * self.stock_dim]
        self.month = 3
        self.start_month = self.month
        self.data = self.data_df.loc[self.month :]
        self.date_memory = [self.data.Date.unique()[0]]
        self.transaction_cost_memory = []

        self.tickers = self.data_df["ticker"].unique().tolist()
        self.others_df = pd.read_csv(other_df_path)
        self.stock1_tic = self.tickers[0]
        self.stock2_tic = self.tickers[1]

        self.stock1_wt = stock1_weight
        self.stock2_wt = stock2_weight
        self.market_caps = {
            self.stock1_tic: self.stock1_wt,
            self.stock2_tic: self.stock2_wt,
        }

        self.stock1_data_close = self.data_df.loc[
            self.data_df["ticker"] == self.stock1_tic
        ]["Adj Close"]
        self.stock2_data_close = self.data_df.loc[
            self.data_df["ticker"] == self.stock2_tic
        ]["Adj Close"]

        self.data_close = pd.DataFrame()
        self.data_close[self.stock1_tic] = self.stock1_data_close.values
        self.data_close[self.stock2_tic] = self.stock2_data_close.values
        self.data_close.reset_index(drop=True, inplace=True)
        self.market_prices = (
            self.stock1_wt * self.stock1_data_close
            + self.stock2_wt * self.stock2_data_close
        )
        self.market_prices.reset_index(drop=True, inplace=True)

    def reset(self, seed=None, options=None):
        self.asset_memory = [self.initial_amount]
        self.month = self.start_month
        self.data = self.data_df.loc[self.month, :]
        self.other_data = self.others_df.loc[self.month, :]

        self.state = [self.data[ic].values.tolist() for ic in self.information_cols]
        self.state.append([self.other_data[oc] for oc in self.other_cols])
        self.state = np.array(self.state)
        self.portfolio_value = self.initial_amount
        self.portfolio_return_memory = [0]

        self.terminal = False
        self.weights_memory = [[self.stock1_wt,self.stock2_wt]]
        self.transaction_cost_memory = []

        return self.state, {}

    def step(self, actions):

        self.terminal = self.month >= len(self.data_df.Date.unique()) - 1
        # print(self.month,len(self.data_df.index)-1)
        actions = np.array(actions)
        self.actions.append(list(actions))
        # print(actions,actions.shape)

        if self.terminal:

            return self.state, self.reward, self.terminal, self.terminal, {}
        else:
            if not self.if_confidence:
                self.stock1_ret = actions[0]
                self.stock2_ret = actions[1]
                self.relative_ret = actions[2]
            else:
                self.stock1_ret = actions[0][0]
                self.stock2_ret = actions[0][1]
                self.relative_ret = actions[0][2]

            # if self.if_confidence:
                self.stock1_conf = actions[1][0]
                self.stock2_conf = actions[1][1]
                self.relative_conf = actions[1][2]
                confidences = [self.stock1_conf, self.stock2_conf, self.relative_conf]

            P_mat = np.array([[1, 0], [0, 1], [1, -1]])

            if self.relative_ret < 0:
                P_mat[2][0] = -1
                P_mat[2][1] = 1
            elif self.relative_ret == 0:
                P_mat[2][0] = 0
                P_mat[2][1] = 0

            Q_mat = np.array(
                [self.stock1_ret, self.stock2_ret, abs(self.relative_ret)]
            ).reshape(-1, 1)

            # print(self.data_close[:1])
            # print(self.data_close[:self.month])
            S = risk_models.CovarianceShrinkage(
                self.data_close[: self.month]
            ).ledoit_wolf()
            # S = risk_models.CovarianceShrinkage(
            #     self.data_close[: self.month]
            # ).shrunk_covariance()
            # S = risk_models.CovarianceShrinkage(self.data_close[:self.month]).oracle_approximating()
            delta = black_litterman.market_implied_risk_aversion(
                self.market_prices[: self.month]
            )

            market_prior = black_litterman.market_implied_prior_returns(
                self.market_caps, delta, S
            )
            # print(market_prior,S,P_mat,Q_mat)

            if self.if_confidence:
                bl = BlackLittermanModel(
                    S,
                    pi=market_prior,
                    P=P_mat,
                    Q=Q_mat,
                    omega="idzorek",
                    view_confidences=confidences,
                )

            else:
                bl = BlackLittermanModel(S, pi=market_prior, P=P_mat, Q=Q_mat)

            ret_bl = bl.bl_returns()

            S_bl = bl.bl_cov()

            try:

                ef = EfficientFrontier(ret_bl, S_bl)
                ef.add_objective(objective_functions.L2_reg)

                ef.max_sharpe(risk_free_rate=self.base_return)
                weights = ef.clean_weights()
            except:
                self.portfolio_value = (
                    self.data["Adj Close"] * self.weights_memory[-1]
                ).values.sum()
                self.asset_memory.append(self.portfolio_value)
                self.weights_memory.append(self.weights_memory[-1])
                self.reward = self.base_return_penalty
                self.portfolio_return_memory.append(
                    -self.base_return_penalty / self.portfolio_value
                )
                self.month += 1

                self.data = self.data_df.loc[self.month, :]
                self.other_data = self.others_df.loc[self.month, :]
                self.state = [self.data[ic].values.tolist() for ic in self.information_cols]
                self.state.append([self.other_data[oc] for oc in self.other_cols])
                self.state = np.array(self.state)

                return self.state, self.reward, self.terminal, self.terminal, {}

            weights = pd.Series(weights).values
            self.weights_memory.append(weights)

            last_day_memory = self.data
            self.month += 1

            self.data = self.data_df.loc[self.month, :]
            self.other_data = self.others_df.loc[self.month, :]
            self.state = [self.data[ic].values.tolist() for ic in self.information_cols]

            self.state.append([self.other_data[oc] for oc in self.other_cols])
            self.state = np.array(self.state)
            portfolio_return = sum(
                (
                    (
                        self.data["Adj Close"].values
                        / last_day_memory["Adj Close"].values
                    )
                    - 1
                )
                * weights
            )
            # new_weights = self.normalization([weights[0]] + list(
            #     np.array(weights[1:]) * np.array(
            #         (self.data["Adj Close"].values / last_day_memory["Adj Close"].values))))
            # self.weights_memory.append(new_weights)

            weights_old = self.weights_memory[-2]
            weights_new = self.weights_memory[-1]

            diff_weights = np.sum(np.abs(np.array(weights_old) - np.array(weights_new)))
            transcationfee = (
                diff_weights * self.transaction_cost_pct * self.portfolio_value
            )

            # calculate the overall result
            new_portfolio_value = (self.portfolio_value - transcationfee) * (
                1 + portfolio_return
            )
            portfolio_return = (
                new_portfolio_value - self.portfolio_value
            ) / self.portfolio_value
            self.reward = new_portfolio_value - self.portfolio_value
            self.portfolio_value = new_portfolio_value

            self.portfolio_return_memory.append(portfolio_return)
            # self.date_memory.append(self.data.Date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            self.reward = self.reward

        return self.state, self.reward, self.terminal, self.terminal, {}

    def save_portfolio_return_memory(self):
        # a record of return for each time stamp
        date_list = self.data_df.Date.unique()[self.start_month :]

        return_list = self.portfolio_return_memory
        df_return = pd.DataFrame(return_list)
        df_return.columns = ["daily_return"]
        df_return.index = date_list

        return df_return

    def normalization(self, actions):
        # a normalization function not only for actions to transfer into weights but also for the weights of the
        # portfolios whose prices have been changed through time
        actions = np.array(actions)
        sum = np.sum(actions)
        actions = actions / sum
        return actions

    def save_asset_memory(self):
        # a record of asset values for each time stamp
        date_list = self.data_df.Date.unique()[self.start_month :]

        assets_list = self.asset_memory
        df_value = pd.DataFrame(assets_list)
        df_value.columns = ["total assets"]
        df_value.index = date_list

        return df_value
