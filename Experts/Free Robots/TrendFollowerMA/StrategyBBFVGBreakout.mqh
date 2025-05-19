//=== StrategyBBFVGBreakout.mqh ===
#ifndef __StrategyBBFVGBreakout_mqh__
#define __StrategyBBFVGBreakout_mqh__

// Fair Value Gap + Support/Resistance Breakout Strategy with Trailing Stop

void EvaluateFVG_Breakout(string sym, SymbolState* state) {
   // Ensure data availability
   if (SeriesInfoInteger(sym, ExpectedTimeframe, SERIES_SYNCHRONIZED) == false)
      return;

   // Exit/Trailing stop management if a position exists
   if (PositionSelect(sym)) {
      ManageTrailingStop(sym, state);
      return;
   }

   // Cooldown to prevent repeated entries
   if (TimeCurrent() - state.lastTradeTime < ReentryCooldownBars * PeriodSeconds(_Period))
      return;

   // Fetch key price points for FVG detection
   double high0 = iHigh(sym, ExpectedTimeframe, 2);
   double low0  = iLow(sym, ExpectedTimeframe, 2);
   double low2  = iLow(sym, ExpectedTimeframe, 0);
   double high2 = iHigh(sym, ExpectedTimeframe, 0);
   double prevClose = iClose(sym, ExpectedTimeframe, 1);
   double currClose = iClose(sym, ExpectedTimeframe, 0);

   // Detect Fair Value Gap (bullish or bearish)
   bool fvgBull = (low2 > high0);
   bool fvgBear = (high2 < low0);
   if (!fvgBull && !fvgBear)
      return;

   // Determine recent support/resistance over SRWindow bars
   double level = fvgBull ? -DBL_MAX : DBL_MAX;
   for (int i = 1; i <= SRWindow; i++) {
      double price = fvgBull ? iHigh(sym, ExpectedTimeframe, i) : iLow(sym, ExpectedTimeframe, i);
      if ((fvgBull && price > level) || (!fvgBull && price < level))
         level = price;
   }

   // Breakout condition
   if (fvgBull) {
      if (!(prevClose <= level && currClose > level))
         return;
   } else {
      if (!(prevClose >= level && currClose < level))
         return;
   }

   // Stop-loss at the FVG candle edge
   double sl = fvgBull ? low2 : high2;

   // Calculate lot size based on ATR risk
   double lot = CalculateLotSize(sym, sl, fvgBull);

   // Open trade with SL only; no fixed TP (trailing stop handles exits)
   OpenTrade(sym, fvgBull, sl, 0.0, lot);
   state.lastTradeTime = TimeCurrent();
}

#endif // __StrategyBBFVGBreakout_mqh__
