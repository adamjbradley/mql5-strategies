//+------------------------------------------------------------------+
//|                                             ROC Rebound EA.mq5   |
//|                       Converted from QuantConnect strategy       |
//+------------------------------------------------------------------+
#property script_show_inputs
input int      ATR_Period = 14;
input int      ROC_Lookback = 14;
input double   ATR_TP_Multiplier = 1.0;
input double   ATR_SL_Multiplier = 2.5;
input double   Trade_Risk_Percent = 1.0;
input double   ROC_Min = -30;
input double   ROC_Max = -15;
input int      Max_Holding_Bars = 15;

//--- Indicator handles
int atr_handle;
double atr_buffer[];

//--- Global variables
double entry_price = 0;
double stop_loss = 0;
double take_profit = 0;
int entry_bar = -1;
bool in_trade = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    atr_handle = iATR(_Symbol, PERIOD_D1, ATR_Period);
    if(atr_handle == INVALID_HANDLE)
    {
        Print("Failed to create ATR handle");
        return INIT_FAILED;
    }
    ArraySetAsSeries(atr_buffer, true);
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Calculate Rate of Change                                         |
//+------------------------------------------------------------------+
double CalculateROC(int lookback)
{
    double price_now = iClose(_Symbol, PERIOD_D1, 0);
    double price_then = iClose(_Symbol, PERIOD_D1, lookback);
    return 100.0 * (price_now - price_then) / price_then;
}

//+------------------------------------------------------------------+
//| Check for long entry condition                                   |
//+------------------------------------------------------------------+
bool ShouldEnterLong()
{
    double roc_today = CalculateROC(ROC_Lookback);
    double roc_yesterday = CalculateROC(ROC_Lookback + 1);
    double roc_3days_ago = CalculateROC(ROC_Lookback + 3);
    return (roc_today >= ROC_Min && roc_today <= ROC_Max &&
            roc_today > roc_yesterday && roc_today > roc_3days_ago);
}

//+------------------------------------------------------------------+
//| Execute trade with ATR-based TP/SL                               |
//+------------------------------------------------------------------+
void EnterTrade()
{
    if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) <= 0) return;
    double atr = atr_buffer[0];
    double lot_size = CalculateLotSize();

    entry_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    stop_loss = entry_price - ATR_SL_Multiplier * atr;
    take_profit = entry_price + ATR_TP_Multiplier * atr;

    MqlTradeRequest request;
    MqlTradeResult result;
    ZeroMemory(request);
    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = lot_size;
    request.type = ORDER_TYPE_BUY;
    request.price = entry_price;
    request.sl = stop_loss;
    request.tp = take_profit;
    request.deviation = 10;
    request.magic = 123456;
    request.type_filling = ORDER_FILLING_IOC;

    OrderSend(request, result);
    if(result.retcode == TRADE_RETCODE_DONE)
    {
        in_trade = true;
        entry_bar = iBarShift(_Symbol, PERIOD_D1, TimeCurrent());
    }
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                 |
//+------------------------------------------------------------------+
double CalculateLotSize()
{
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amount = balance * Trade_Risk_Percent / 100.0;
    double sl_pips = (ATR_SL_Multiplier * atr_buffer[0]) / SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double lot = risk_amount / (sl_pips * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE));
    return NormalizeDouble(lot, 2);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    if(in_trade)
    {
        int current_bar = iBarShift(_Symbol, PERIOD_D1, TimeCurrent());
        if((entry_bar - current_bar) > Max_Holding_Bars)
        {
            // Close trade
            double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            MqlTradeRequest request;
            MqlTradeResult result;
            ZeroMemory(request);
            request.action = TRADE_ACTION_DEAL;
            request.symbol = _Symbol;
            request.volume = PositionGetDouble(POSITION_VOLUME);
            request.type = ORDER_TYPE_SELL;
            request.price = price;
            request.deviation = 10;
            request.magic = 123456;
            request.type_filling = ORDER_FILLING_IOC;
            OrderSend(request, result);
            in_trade = false;
        }
        return;
    }

    if(ShouldEnterLong())
        EnterTrade();
}
