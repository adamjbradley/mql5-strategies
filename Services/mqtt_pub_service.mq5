//+------------------------------------------------------------------+
//|                                             mqtt_pub_service.mq5 |
//|            ********* WORK IN PROGRESS **********                 |
//| **** PART OF ARTICLE https://www.mql5.com/en/articles/14677 **** |
//+------------------------------------------------------------------+
#include <MQTT\MQTT.mqh>
#include <MQTT\Connect.mqh>
#include <MQTT\Publish.mqh>

#include <JAson.mqh>
#include <Base64.mqh>
#include <Generic\HashMap.mqh>

#property service

//--- input parameters
//input string host = "100.118.216.62";
input string host = "192.168.1.102";
input int      port = 1883;
input string   symbol = "EURUSD";
input int      frequency = 1000;
input string   current_version = "1.1";

input bool     backtest = true;
input string   _start_date = "2024-01-01";
input string   _end_date = "2024-02-01";

input ENUM_TIMEFRAMES   timeframe = PERIOD_M1;

input string   client_identifier ="MQTT Pub Service";

input string   publish_topic = "/";
input string   subscribe_topic = "/orders";

//---
int skt;
CConnect *conn;
CPublish *pub;

CHashMap <string, long> last_tick_per_symbol;


datetime start_date;
datetime end_date;



//+------------------------------------------------------------------+
//| Service program start function                                   |
//+------------------------------------------------------------------+
int OnStart()
  {
   Print(__FILE__ + " : " + __FUNCTION__);
   Print("MQTT Publish Service started");
//---
   uchar conn_pkt[];
   conn = new CConnect(host, port);
   conn.SetCleanStart(true);
   conn.SetKeepAlive(3600);
   conn.SetClientIdentifier("MT5_PUB");
   conn.Build(conn_pkt);
   ArrayPrint(conn_pkt);
   
   start_date = _start_date == "Today" ? TimeLocal() : StringToTime(_start_date);
   end_date = _end_date == "Today" ? TimeLocal() : StringToTime(_end_date);
   
   int index = 0;
   
//---
   if(SendConnect(host, port, conn_pkt) == 0)
     {
      Print("Client connected ", host);
     }
 
     string payload = NULL;
  
 
     if (backtest) {
      MqlRates rates[];   
      int copied = CopyRates(symbol, timeframe, start_date, end_date, rates);
      if(copied<=0)
         Print("Error copying price data ",GetLastError());
      else
         Print("Copied ", ArraySize(rates), " bars");
      }
      
      else {
 
         do
           {              
            payload = GetRates();
            //string payload = GetLastTick();
            
            //pub.SetContentType("application/json");
            //string base64payload;
            //base64payload = Base64Encode(payload);
            //Print(base64payload);
         
         
         if (payload != NULL)
         {
            uchar pub_pkt[];
            pub = new CPublish();
            pub.SetTopicName("/");
            
            pub.SetPayload(payload);
            pub.Build(pub_pkt);
         
            delete(pub);
            
            //ArrayPrint(pub_pkt);
            if(!SendPublish(pub_pkt))
              {
               return -1;
               CleanUp();
              }
            ZeroMemory(pub_pkt);
            Sleep(frequency);
         }
        }
      while(!IsStopped());
     }
//---
   CleanUp();
   return 0;
  }
    
  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string GetLastTickJSON()
  {
   MqlTick last_tick;
   if(SymbolInfoTick(symbol, last_tick))
     {
      
      string format = "%G-%G-%G-%d-%I64d-%d-%G";
     
      // TODO serialise SymbolInfoTick to JSON
      string json_out;
      CJAVal json(NULL,jtUNDEF);
      json["symbol"] = symbol;
      json["time"] = TimeToString(last_tick.time, TIME_SECONDS);
      json["bid"] = last_tick.bid;
      json["ask"] = last_tick.ask;
      json["last"] = (string) last_tick.last;
      json["volume"] = (string) last_tick.volume;
      json["time_msc"] = last_tick.time_msc;
      json["flags"] = (string) last_tick.flags;
      json["volume_real"] = last_tick.volume_real;
      json.Serialize(json_out); // serialized
      Print(json_out);
      return json_out;
     }
   else
      Print("Failed to get rates for ", symbol);
   return "";
  }
  
  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string GetLastTick()
  {
   MqlTick last_tick;
   if(SymbolInfoTick(symbol, last_tick))
     {
      string format = "%G-%G-%G-%d-%I64d-%d-%G";
      string out;
      
      out = TimeToString(last_tick.time, TIME_SECONDS);
      out += "-" + symbol + "-" + current_version + "-" + StringFormat(format,
          
                                last_tick.bid, //double
                                last_tick.ask, //double
                                last_tick.last, //double
                                last_tick.volume, //ulong
                                last_tick.time_msc, //long
                                last_tick.flags, //uint
                                last_tick.volume_real);//double
                                     
      long time_msc_out = NULL;
      last_tick_per_symbol.TryGetValue(symbol, time_msc_out);
          
      if (last_tick.time_msc == time_msc_out) {
         return NULL;
      }
      else {
         last_tick_per_symbol.TrySetValue(symbol, last_tick.time_msc);
         
         Print(last_tick.time,
            " Symbol = ", symbol,
            " Current Version = ", current_version,
            " Bid = ", last_tick.bid,
            " Ask = ", last_tick.ask,
            " Last = ", last_tick.last,
            " Volume = ", last_tick.volume,
            " Time msc = ", last_tick.time_msc,
            " Flags = ", last_tick.flags,
            " Vol Real = ", last_tick.volume_real
           );
         Print(out);        
         return out;
      }
     }
   else
      Print("Failed to get rates for ", symbol);
   return "";
  }
 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string GetRates()
  {
   MqlRates rates[];
   int copied = CopyRates(symbol, PERIOD_M1, 0, 1, rates);
   if(copied > 0)
     {
      string format = "%G-%G-%G-%G-%d-%d";
      string out;
      out = TimeToString(rates[0].time);
      out += "-" + symbol + "-" + current_version + "-" + StringFormat(format,
                                rates[0].open,
                                rates[0].high,
                                rates[0].low,
                                rates[0].close,
                                rates[0].tick_volume,
                                rates[0].real_volume);
      Print(out);
      return out;
     }
   else
      Print("Failed to get rates for ", symbol);
   return "";
  }
  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string GetRatesByDate()
  {
   MqlRates rates[];
   
   int copied = CopyRates(symbol, PERIOD_M1, start_date, end_date, rates);
   if(copied > 0)
     {
      string format = "%G-%G-%G-%G-%d-%d";
      string out;
      out = TimeToString(rates[0].time);
      out += "-" + symbol + "-" + current_version + "-" + StringFormat(format,
                                rates[0].open,
                                rates[0].high,
                                rates[0].low,
                                rates[0].close,
                                rates[0].tick_volume,
                                rates[0].real_volume);
      Print(out);
      return out;
     }
   else
      Print("Failed to get rates for ", symbol);
   return "";
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SendPublish(uchar & pkt[])
  {
   if(skt == INVALID_HANDLE || SocketSend(skt, pkt, ArraySize(pkt)) < 0)
     {
      Print("Failed sending publish ", GetLastError());
      CleanUp();
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int SendConnect(const string h, const int p, uchar & pkt[])
  {
   skt = SocketCreate();
   if(skt != INVALID_HANDLE)
     {
      if(SocketConnect(skt, h, p, 1000))
        {
         Print("Socket Connected ", h);
        }
     }
   if(SocketSend(skt, pkt, ArraySize(pkt)) < 0)
     {
      Print("Failed sending connect ", GetLastError());
      CleanUp();
     }
//---
   char rsp[];
   SocketRead(skt, rsp, 4, 1000);
   if(rsp[0] >> 4 != CONNACK)
     {
      Print("Not Connect acknowledgment");
      CleanUp();
      return -1;
     }
   if(rsp[3] != MQTT_REASON_CODE_SUCCESS)  // Connect Return code (Connection accepted)
     {
      Print("Connection Refused");
      CleanUp();
      return -1;
     }
   ArrayPrint(rsp);
   return 0;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CleanUp()
  {
   delete pub;
   delete conn;
   SocketClose(skt);
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
