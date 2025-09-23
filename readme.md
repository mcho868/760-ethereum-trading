Potential Problems (ethTradingDrlAgent.ipynb)

  1. Action & Risk Shaping Consistency (Position Dynamics + Reward/Risk Design)
  2. 
    (1) Per-step position shift constraint is not enforced (agent can jump from −1 to +1 in one step).
        The env comments say “Apply position shift constraint,” but the code only clips the raw action to [-1,1]. It does not cap the change in position between steps.
        Why it matters: The agent can “teleport” from −1 to +1 in one step, creating unrealistic turnover, excessive costs, and unstable learning.
     
    (2) Reward scale is imbalanced: action-shaping terms dominate PnL, driving over-trading.
        Issue: activity_reward (0.1) fires on small changes and often exceeds typical per-step PnL signal, pushing the agent to over-trade.
        Why it matters: The agent learns to “move” rather than to make money; this matches the observed instability after adding sentiment.
        
    (3) Drawdown penalty is discrete and too large, creating spiky rewards.
        Issue: Fixed −50 once a threshold (e.g., 10%) is hit—sparse and abrupt.
        Why it matters: Spiky, rare penalties push the policy into odd local minima.
  
  2. Agent holds positions that are too small during evaluation
     
     (1)Action → Position pipeline
        max_position_shift is too low, so position can’t ramp up quickly.
        Post-processing double-clips/compresses actions; effective Δpos becomes tiny.
        Ambiguity between “target position” vs “incremental change” further reduces realized position.
        Any “min trade threshold” swallows small adjustments, preventing accumulation.
     
     (2)Reward & risk shaping
        Action-shaping terms outweigh PnL, incentivizing small/neutral positions.
        Discrete, large drawdown penalties make the policy avoid higher exposure.
        Costs/slippage (or duplicated cost penalties) discourage scaling up.
     
     (3)Evaluation flow / API usage
        Gym vs. Gymnasium (and VecEnv) mismatch in the evaluation loop leads to premature terminations or muted outputs.
        Non-deterministic prediction or too-short evaluation windows prevent position ramp-up.
     
     (4)Observation & normalization
        Normalization fitted on full data (not train-only) and not re-used at eval → distribution shift → outputs gravitate toward zero.
        Observation space bounds don’t match feature scaling, causing constant clipping.
        Sentiment/auxiliary features are noisy or mis-timed (no smoothing/lag), pushing the policy toward conservative actions.
     
  3. Trading cost redesign
     
      (1)Is the cost definition unambiguous: single-sided or round-trip? Charged in bps on notional, or a fixed fee per traded value?
         Touchpoints: REWARD_COMPONENTS (defaults like fee_rate, slippage) and the specific lines in the environment where trading costs are computed (see B-1).
     
      (2)Are “spread” and “slippage” being conflated? Are we only using a bps fee and forgetting a constant spread component?
         Touchpoints: Currently only fee_rate/slippage scalars are present; we should add a separate spread_bps (or proxy it via high–low).
     
      (3)Are costs being double-counted (once in portfolio updates and again in the reward channel)?
         Touchpoints: Portfolio value update vs. the transaction_penalty reward term (even if its weight is 0 now, eliminate the possibility of duplication).
     
      (4)Are costs correctly linked to turnover (|Δposition|)? Are the units consistent (per minute vs. per trade)?
         Touchpoints: Verify that the position_change used in cost computation reflects actual traded volume, and that the fee multiplies |Δpos| * notional.
     
      (5)Do we model slippage as a function of volatility/volume (higher volatility or larger size → higher slippage)?
         Touchpoints: Consider a dynamic slippage term (e.g., based on ATR% and/or trade size) rather than a single static scalar.

  4. Reward function imbalance — PnL signal is too quiet, side terms dominate, strategy optimizes constraints instead of returns.

      REWARD_COMPONENTS: pnl_scale is too small for minute-level returns, while activity_reward=0.1, inactivity_penalty=0.005, sharpe_weight are large enough to outweigh PnL.
      EthereumTradingEnvironment._calculate_multi_component_reward(...): side channels (activity/holding/sharpe/drawdown) are added at full strength every step; drawdown is a discrete big penalty; sentiment reward can be mis-scaled if enabled.
      Observation scaling/Box bounds can clip features, indirectly shrinking PnL variability seen by the policy (keeps actions small → small PnL reward).
     
     How do we make PnL the loudest signal in training so the agent maximizes returns rather than “satisfying constraints”?
      Do we need to increase pnl_scale to the correct order of magnitude for minute data?
      Should we turn down action-shaping terms (activity_reward, inactivity_penalty, sharpe_weight) and replace discrete drawdown with a smooth penalty?
      Do we have any hidden inflators (e.g., sentiment reward scaled by 1/max_position_shift) that hijack the objective?
      Are feature/Box ranges causing clipping that suppresses the PnL signal the network can learn from?

  5. Redesign a Trend filter
     
     We have incorporated many mean-reversion features into the state space—for example, z-score and zone.
     However, should we design a trend filter to determine when the market is in a ranging regime—where mean-reversion is appropriate—and when it is trending, in which case we should stop mean-reversion or switch strategies?
     I previously tried to design this filter using ATR, and I also used EMA as a second safeguard.
     You can follow this approach to design it, or use your own method.
     
  6. Sentiment Weighting
     
     We use five small models, but we currently take the simple average of their scores.
     This creates a problem because the five models often disagree—some output positive values while others output negative values.
     If we just average them, the result may become insignificant.（For example, if two models output +1 and two output −1, the mean becomes 0.）
     Is there a better weighting scheme than a plain average?
     Another question: if we’re already using five small language models, do we still need VADER scores?(We actually have VADER scores in our dataset.)
     Which approach performs better in practice?（Or could we combine them and use them together?）

 7. whether 15D is truly better than 5D—this needs continual testing and validation
    
    First, if the current price range is merely ranging (the sweet spot for mean reversion), then beyond using mean-reversion features like z-score and zone, do we actually need to add other technical indicators?
    Do those indicators genuinely help, or do they just introduce noise?
    If we are in a trending regime, do mean-reversion features still help at all?
