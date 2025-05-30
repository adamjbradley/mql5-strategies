#property indicator_separate_window
#property indicator_buffers 1
#property indicator_plots   1

#property indicator_label1  "LWTI"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_width1  2

//--- input parameters
input int LWTIPeriod    = 25;   // Raw volume EMA period
input int LWTISmoothing = 20;   // EMA smoothing period

//--- indicator buffers
double rawVol[];
double emaRaw[];
double LWTIBuffer[];

int OnInit()
{
    //--- indicator buffers mapping
    SetIndexBuffer(0, LWTIBuffer, INDICATOR_DATA);

    //--- set as series (0 = current bar)
    ArraySetAsSeries(rawVol, true);
    ArraySetAsSeries(emaRaw, true);
    ArraySetAsSeries(LWTIBuffer, true);

    //--- indicator name
    IndicatorSetString(INDICATOR_SHORTNAME, "Larry Williams Trade Indicator");
    
    //--- name for DataWindow
    PlotIndexSetString(0, PLOT_LABEL, "LWTI");
    IndicatorSetInteger(INDICATOR_DIGITS, _Digits);

    return(INIT_SUCCEEDED);
}

int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
    //--- ensure internal arrays are sized
    if(ArraySize(rawVol)!=rates_total)
        ArrayResize(rawVol, rates_total);
    if(ArraySize(emaRaw)!=rates_total)
        ArrayResize(emaRaw, rates_total);

    int start = prev_calculated;
    if(start < 1) start = 1; // start from second bar for recursion

    double alpha1 = 2.0 / (LWTIPeriod + 1.0);
    double alpha2 = 2.0 / (LWTISmoothing + 1.0);

    for(int i = start; i < rates_total; i++)
    {
        //--- raw volume weighted sign
        rawVol[i] = (close[i] > open[i]) ? tick_volume[i] : 
                    ((close[i] < open[i]) ? -tick_volume[i] : 0);
        //--- first EMA on rawVol
        if(i == start)  // initialize first EMA value
            emaRaw[i-1] = rawVol[i-1];
        emaRaw[i] = rawVol[i] * alpha1 + emaRaw[i-1] * (1 - alpha1);
        //--- second EMA gives LWTI
        if(i == start)
            LWTIBuffer[i-1] = emaRaw[i-1];
        LWTIBuffer[i] = emaRaw[i] * alpha2 + LWTIBuffer[i-1] * (1 - alpha2);
    }

    return(rates_total);
}
