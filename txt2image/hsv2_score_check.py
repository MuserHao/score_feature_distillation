# res = {
#  "h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w11z_w10.005t_w21z_w20.005": ["0.262", "0.3157", "0.2352", "0.2976", "0.2402", "0.1995", "0.2135", "0.2274", "0.2245", "0.2617"], 
#  "h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.005t_w21z_w20.005": ["0.262", "0.2842", "0.2445", "0.2717", "0.2137", "0.2634", "0.2566", "0.24", "0.2462", "0.1917"], 
#  "h512w512timeline-1":                                                                      ["0.1665", "0.2129", "0.2407", "0.268", "0.2878", "0.2515", "0.2788", "0.2412", "0.2454", "0.269"], 
#  "h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.005t_w210z_w20.002": ["0.243", "0.2566", "0.2334", "0.2236", "0.2634", "0.19", "0.2656", "0.2286", "0.2415", "0.2477"], 
#  "h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.005t_w21z_w20.002": ["0.2367", "0.2986", "0.1815", "0.2964", "0.171", "0.2251", "0.2319", "0.226", "0.2161", "0.265"], 
#  "h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.002t_w210z_w20.002": ["0.2389", "0.2546", "0.2269", "0.15", "0.2593", "0.1564", "0.2544", "0.2037", "0.2418", "0.2798"], 
#  "h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.005t_w210z_w20.005": ["0.1595", "0.1998", "0.1754", "0.2329", "0.2135", "0.1525", "0.2598", "0.1735", "0.2279", "0.2208"],
#  "h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w11z_w10.005t_w21z_w20.005": ["0.2314", "0.2433", "0.2147", "0.2642", "0.2747", "0.2725", "0.2893", "0.2045", "0.244", "0.2433"], 
#  "h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Truet_w11z_w10.005t_w21z_w20.005": ["0.2607", "0.2461", "0.2644", "0.2676", "0.2747", "0.2456", "0.2842", "0.1967", "0.2395", "0.2316"],
#  "h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.005t_w21z_w20.005": ["0.1606", "0.1982", "0.2223", "0.2527", "0.2642", "0.2314", "0.2832", "0.2306", "0.2286", "0.2427"],
#  "h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.005t_w210z_w20.005": ["0.2607", "0.1725", "0.2312", "0.1613", "0.273", "0.2178", "0.2834", "0.2141", "0.2466", "0.275"], 
#  "h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.002t_w210z_w20.002": ["0.2637", "0.2155", "0.1934", "0.1912", "0.2388", "0.1482", "0.2537", "0.2151", "0.2285", "0.271"], 
#  "h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Falset_w11z_w10.005t_w21z_w20.005": ["0.2754", "0.2235", "0.2236", "0.2408", "0.258", "0.1323", "0.2578", "0.2383", "0.2467", "0.302"],
#  "h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.005t_w210z_w20.002": ["0.2532", "0.2527", "0.2228", "0.2476", "0.2433", "0.2222", "0.279", "0.1963", "0.2344", "0.2769"], 
#  "h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.005t_w21z_w20.002": ["0.2234", "0.2322", "0.2495", "0.1989", "0.2705", "0.1659", "0.2607", "0.2115", "0.2388", "0.2595"]}

# import numpy as np
# for k, v in res.items():
#     vv = [float(x) for x in v]
#     print(k, np.mean(vv))
    
import json

with open('hpsv2_scores.json') as f:
    data = json.loads(f.read())
# data = json.loads('hpsv2_scores.json')
import numpy as np
for k, v in data.items():
    vv = [float(x) for x in v]
    print(k, np.mean(vv))
    
"""
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w11z_w10.005t_w21z_w20.005 0.24772999999999995
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Falset_w11z_w10.005t_w21z_w20.005 0.24791000000000002
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Falset_w11z_w1-0.005t_w21z_w2-0.005 0.23275
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w11z_w1-0.005t_w210z_w2-0.005 0.23600999999999997

h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w1-0.005t_w210z_w2-0.002 0.25831
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w11z_w10.005t_w21z_w20.005 0.24819
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w11z_w1-0.005t_w21z_w2-0.005 0.24073000000000003
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Truet_w11z_w10.005t_w21z_w20.005 0.25111




h512w512timeline-1 0.24618
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.005t_w21z_w20.005 0.2474
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w1-0.005t_w210z_w2-0.005 0.24512
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w1-0.005t_w21z_w2-0.002 0.25123
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.005t_w210z_w20.002 0.23933999999999997
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.005t_w21z_w20.002 0.23483
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.002t_w210z_w20.002 0.22658
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.005t_w210z_w20.005 0.20156000000000002

h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.005t_w21z_w20.005 0.23145000000000002
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.005t_w210z_w20.005 0.23356
h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.002t_w210z_w20.002 0.22190999999999997


h512w512timeline200extrap1Trueextrap2Truepos1Falsepos2Truet_w110z_w10.005t_w210z_w20.002 0.24284
h512w512timeline200extrap1Trueextrap2Truepos1Truepos2Falset_w110z_w10.005t_w21z_w20.002 0.23109000000000002

"""