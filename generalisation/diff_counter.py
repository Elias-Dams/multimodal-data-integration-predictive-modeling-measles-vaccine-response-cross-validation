import matplotlib.pyplot as plt

########## RNA UNCOMPRESSED ##########

# title = "Feature Importance Trajectories: RNA"

# dicts = {
#     "100": {
#         "TBD.25": 0.3376,
#         "Inflammation.6": 0.2583,
#         "TBD.79": 0.2523,
#         "Neutrophils.2": 0.2515,
#         "Inflammation": 0.2395,
#         "TBD.152": 0.1879,
#         "TBD.167": 0.1870,
#         "B cells": 0.1707,
#         "Protein synthesis.17": 0.1571,
#         "TBD.172": 0.1412,
#         "Monocytes.3": 0.1396,
#         "Inflammation.5": 0.0820,
#         "Leukocyte activation": 0.0717,
#         "Erythroid cells.14": 0.0701,
#         "Cell cycle.5": 0.0635,
#         "B cells.2": 0.0551,
#         "Platelet.1": 0.0267,
#         "Platelet.3": 0.0229,
#         "Erythroid cells.12": 0.0165,
#         "Inflammation.3": 0.0077,
#         "Cell cycle": 0.0000,
#         "TBD": 0.0000,
#         "Prostanoids": 0.0000,
#         "Type 1 Interferon": 0.0000,
#         "Cytotoxic lymphocytes": 0.0000,
#         "Erythroid cells": 0.0000,
#         "Interferon": 0.0000,
#         "Protein synthesis": 0.0000,
#         "Platelet": 0.0000,
#         "Neutrophil activation": 0.0000
#     },
#     "200": {
#         "Inflammation.6": 0.2934,
#         "Neutrophils.2": 0.2317,
#         "Erythroid cells.14": 0.2264,
#         "Inflammation": 0.2188,
#         "Protein synthesis.17": 0.2054,
#         "Monocytes.3": 0.1469,
#         "Leukocyte activation": 0.1237,
#         "Inflammation.5": 0.1158,
#         "Inflammation.3": 0.0950,
#         "Inflammation.1": 0.0827,
#         "Monocytes.4": 0.0717,
#         "Erythroid cells.11": 0.0703,
#         "Platelet/Prostaglandin": 0.0633,
#         "Neutrophils": 0.0619,
#         "Platelet.3": 0.0603,
#         "Cytokines/chemokines.2": 0.0549,
#         "Lymphocytes.1": 0.0549,
#         "Lymphocytes.3": 0.0424,
#         "Cell cycle.5": 0.0395,
#         "Cytotoxic lymphocytes": 0.0364,
#         "Erythroid cells.7": 0.0348,
#         "Platelet.1": 0.0336,
#         "Platelet": 0.0283,
#         "Inflammation.4": 0.0281,
#         "Gene transcription.17": 0.0265,
#         "Erythroid cells.17": 0.0221,
#         "Neutrophil activation": 0.0078,
#         "Cell cycle": 0.0000,
#         "Prostanoids": 0.0000,
#         "Type 1 Interferon": 0.0000
#     },
#     "300": {
#         "Inflammation.6": 0.2948,
#         "Neutrophils.2": 0.2334,
#         "Inflammation": 0.2190,
#         "Erythroid cells.14": 0.2184,
#         "Protein synthesis.17": 0.2055,
#         "Monocytes.3": 0.1546,
#         "Inflammation.5": 0.1098,
#         "Leukocyte activation": 0.1086,
#         "Inflammation.3": 0.1037,
#         "Inflammation.1": 0.0911,
#         "Neutrophils": 0.0628,
#         "Erythroid cells.11": 0.0627,
#         "Platelet.3": 0.0560,
#         "Cytokines/chemokines.2": 0.0542,
#         "Platelet/Prostaglandin": 0.0529,
#         "Monocytes.4": 0.0513,
#         "Lymphocytes.1": 0.0458,
#         "Lymphocytes.3": 0.0429,
#         "Erythroid cells.7": 0.0337,
#         "Cell cycle.5": 0.0304,
#         "B cells": 0.0295,
#         "Platelet.1": 0.0258,
#         "Platelet": 0.0251,
#         "Cytotoxic lymphocytes": 0.0243,
#         "Gene transcription.17": 0.0197,
#         "Inflammation.4": 0.0149,
#         "Erythroid cells.17": 0.0120,
#         "Neutrophil activation": 0.0046,
#         "Cell cycle": 0.0000,
#         "Prostanoids": 0.0000
#     },
#     "400": {
#         "Inflammation.6": 0.2977,
#         "Neutrophils.2": 0.2367,
#         "Inflammation": 0.2245,
#         "Protein synthesis.17": 0.1964,
#         "Erythroid cells.14": 0.1889,
#         "Monocytes.3": 0.1576,
#         "Leukocyte activation": 0.1160,
#         "Inflammation.5": 0.1122,
#         "Inflammation.3": 0.1114,
#         "Inflammation.1": 0.0948,
#         "Neutrophils": 0.0627,
#         "Erythroid cells.11": 0.0560,
#         "Platelet.3": 0.0528,
#         "Cytokines/chemokines.2": 0.0509,
#         "Monocytes.4": 0.0496,
#         "Platelet/Prostaglandin": 0.0460,
#         "Lymphocytes.1": 0.0431,
#         "Cell cycle.5": 0.0431,
#         "Lymphocytes.3": 0.0402,
#         "Erythroid cells.7": 0.0337,
#         "Platelet.1": 0.0305,
#         "B cells": 0.0268,
#         "Platelet": 0.0250,
#         "Inflammation.4": 0.0230,
#         "Cytotoxic lymphocytes": 0.0132,
#         "Erythroid cells.17": 0.0114,
#         "Gene transcription.17": 0.0097,
#         "Cell cycle": 0.0000,
#         "Prostanoids": 0.0000,
#         "Type 1 Interferon": 0.0000
#     },
#     "500": {
#         "Inflammation.6": 0.3011,
#         "Neutrophils.2": 0.2469,
#         "Inflammation": 0.2356,
#         "Protein synthesis.17": 0.2046,
#         "Erythroid cells.14": 0.1887,
#         "Monocytes.3": 0.1600,
#         "Leukocyte activation": 0.1248,
#         "Inflammation.5": 0.1156,
#         "Inflammation.3": 0.1152,
#         "Inflammation.1": 0.1001,
#         "Neutrophils": 0.0653,
#         "Erythroid cells.11": 0.0600,
#         "Platelet.3": 0.0585,
#         "Cytokines/chemokines.2": 0.0515,
#         "Monocytes.4": 0.0480,
#         "Platelet/Prostaglandin": 0.0460,
#         "Platelet.1": 0.0436,
#         "Lymphocytes.3": 0.0423,
#         "Lymphocytes.1": 0.0390,
#         "Inflammation.4": 0.0374,
#         "Erythroid cells.7": 0.0359,
#         "Cell cycle.5": 0.0357,
#         "B cells": 0.0314,
#         "Platelet": 0.0238,
#         "Cytotoxic lymphocytes": 0.0165,
#         "Erythroid cells.17": 0.0123,
#         "Gene transcription.17": 0.0096,
#         "Erythroid cells.3": 0.0010,
#         "Cell cycle": 0.0000,
#         "Prostanoids": 0.0000
#     }
# }

######################################

####### CYTOKINES UNCOMPRESSED #######

# title = "Feature Importance Trajectories: cytokines"

# dicts = {
#     "100": {
#         "HHV6.Status": 0.2434,
#         "IL17A": 0.1442,
#         "Flt3 Ligand": 0.1167,
#         "VEGF": 0.1079,
#         "GRO": 0.0993,
#         "IL-8": 0.0868,
#         "SCD40L": 0.0814,
#         "MDC": 0.0747,
#         "IL-2": 0.0728,
#         "GM-CSF": 0.0423,
#         "IL-6": 0.0405,
#         "IL-7": 0.0355,
#         "TNFa": 0.0301,
#         "IL-15": 0.0231,
#         "IL-13": 0.0184,
#         "EGF": 0.0161,
#         "TNFb": 0.0070,
#         "IFNa2": 0.0066,
#         "CMV.Status": 0.0062,
#         "EBV.Status": 0.0000,
#         "HSV1_2.Status": 0.0000,
#         "FGF-2": 0.0000,
#         "Eotaxin": 0.0000,
#         "TGF-a": 0.0000,
#         "GCSF": 0.0000,
#         "Fractalkine": 0.0000,
#         "IFNg": 0.0000,
#         "IL-10": 0.0000,
#         "MCP3": 0.0000,
#         "IL12-p40": 0.0000
#     },
#     "200":  {
#         "GRO": 0.2084,
#         "HHV6.Status": 0.2056,
#         "IL-8": 0.1765,
#         "IL17A": 0.0902,
#         "VEGF": 0.0817,
#         "Flt3 Ligand": 0.0713,
#         "IL-2": 0.0644,
#         "IL-7": 0.0423,
#         "IL-15": 0.0403,
#         "IP-10": 0.0279,
#         "GM-CSF": 0.0262,
#         "MDC": 0.0204,
#         "GCSF": 0.0123,
#         "IL-6": 0.0075,
#         "IL-13": 0.0050,
#         "IFNa2": 0.0033,
#         "CMV.Status": 0.0031,
#         "Fractalkine": 0.0008,
#         "EBV.Status": 0.0000,
#         "HSV1_2.Status": 0.0000,
#         "EGF": 0.0000,
#         "FGF-2": 0.0000,
#         "Eotaxin": 0.0000,
#         "TGF-a": 0.0000,
#         "IFNg": 0.0000,
#         "IL-10": 0.0000,
#         "MCP3": 0.0000,
#         "IL12-p40": 0.0000,
#         "IL12-p70": 0.0000,
#         "SCD40L": 0.0000
#     },
#     "300": {
#         "HHV6.Status": 0.2834,
#         "IL-9": 0.1261,
#         "GRO": 0.0970,
#         "IL-7": 0.0843,
#         "IL-15": 0.0646,
#         "IL-2": 0.0566,
#         "IL-8": 0.0507,
#         "IL17A": 0.0436,
#         "GM-CSF": 0.0430,
#         "VEGF": 0.0336,
#         "IL-13": 0.0174,
#         "IL12-p40": 0.0170,
#         "GCSF": 0.0150,
#         "Flt3 Ligand": 0.0095,
#         "IFNa2": 0.0046,
#         "IL-10": 0.0044,
#         "IL1a": 0.0025,
#         "IP-10": 0.0008,
#         "CMV.Status": 0.0000,
#         "EBV.Status": 0.0000,
#         "HSV1_2.Status": 0.0000,
#         "EGF": 0.0000,
#         "FGF-2": 0.0000,
#         "Eotaxin": 0.0000,
#         "TGF-a": 0.0000,
#         "Fractalkine": 0.0000,
#         "IFNg": 0.0000,
#         "MCP3": 0.0000,
#         "MDC": 0.0000,
#         "IL12-p70": 0.0000
#     },
#     "400": {
#         "HHV6.Status": 0.2082,
#         "GRO": 0.1159,
#         "IL-9": 0.1123,
#         "IL-7": 0.0971,
#         "IL-15": 0.0656,
#         "IL-2": 0.0576,
#         "IL17A": 0.0439,
#         "GM-CSF": 0.0423,
#         "VEGF": 0.0328,
#         "IL-8": 0.0316,
#         "IL-1b": 0.0310,
#         "IL-10": 0.0293,
#         "IL-5": 0.0278,
#         "IL12-p40": 0.0211,
#         "IL-13": 0.0146,
#         "Flt3 Ligand": 0.0061,
#         "IL1a": 0.0038,
#         "GCSF": 0.0034,
#         "CMV.Status": 0.0000,
#         "EBV.Status": 0.0000,
#         "HSV1_2.Status": 0.0000,
#         "EGF": 0.0000,
#         "FGF-2": 0.0000,
#         "Eotaxin": 0.0000,
#         "TGF-a": 0.0000,
#         "Fractalkine": 0.0000,
#         "IFNa2": 0.0000,
#         "IFNg": 0.0000,
#         "MCP3": 0.0000,
#         "MDC": 0.0000
#     },
#     "500": {
#         "HHV6.Status": 0.2085,
#         "IL-9": 0.1287,
#         "IL-7": 0.1126,
#         "GRO": 0.0848,
#         "IL-15": 0.0846,
#         "IL-2": 0.0553,
#         "IL-10": 0.0486,
#         "IL-8": 0.0447,
#         "GM-CSF": 0.0438,
#         "IL-5": 0.0420,
#         "IL17A": 0.0392,
#         "IL-1b": 0.0381,
#         "IL12-p40": 0.0211,
#         "IL-13": 0.0130,
#         "GCSF": 0.0116,
#         "IFNa2": 0.0115,
#         "VEGF": 0.0112,
#         "TNFb": 0.0098,
#         "IL1a": 0.0068,
#         "Flt3 Ligand": 0.0027,
#         "IP-10": 0.0004,
#         "CMV.Status": 0.0000,
#         "EBV.Status": 0.0000,
#         "HSV1_2.Status": 0.0000,
#         "EGF": 0.0000,
#         "FGF-2": 0.0000,
#         "Eotaxin": 0.0000,
#         "TGF-a": 0.0000,
#         "Fractalkine": 0.0000,
#         "IFNg": 0.0000
#     }
# }

######################################

####### CYTOMETRY UNCOMPRESSED #######

# title = "Feature Importance Trajectories: cytometry"
#
# dicts = {
#     "100": {
#         "HGB Day 0": 0.5834,
#         "WBC Day 0": 0.5257,
#         "%LYM Day 0": 0.5123,
#         "PLT Day 0": 0.4792,
#         "%GRA Day 0": 0.4279,
#         "HCT Day 0": 0.3660,
#         "RBC Day 0": 0.3576,
#         "%MON Day 0": 0.2894
#     },
#     "200": {
#         "HGB Day 0": 0.6106,
#         "PLT Day 0": 0.5388,
#         "%LYM Day 0": 0.5000,
#         "WBC Day 0": 0.4989,
#         "%GRA Day 0": 0.3947,
#         "RBC Day 0": 0.3606,
#         "HCT Day 0": 0.3518,
#         "%MON Day 0": 0.2782
#     },
#     "300": {
#         "HGB Day 0": 0.6283,
#         "PLT Day 0": 0.5209,
#         "%LYM Day 0": 0.5121,
#         "WBC Day 0": 0.5068,
#         "%GRA Day 0": 0.4023,
#         "HCT Day 0": 0.3623,
#         "RBC Day 0": 0.3586,
#         "%MON Day 0": 0.2705
#     },
#     "400": {
#         "HGB Day 0": 0.6283,
#         "%LYM Day 0": 0.5235,
#         "PLT Day 0": 0.5209,
#         "WBC Day 0": 0.5105,
#         "%GRA Day 0": 0.4052,
#         "HCT Day 0": 0.3708,
#         "RBC Day 0": 0.3680,
#         "%MON Day 0": 0.2698
#     },
#     "500": {
#         "HGB Day 0": 0.6310,
#         "%LYM Day 0": 0.5280,
#         "PLT Day 0": 0.5244,
#         "WBC Day 0": 0.5118,
#         "%GRA Day 0": 0.4013,
#         "HCT Day 0": 0.3721,
#         "RBC Day 0": 0.3688,
#         "%MON Day 0": 0.2667
#     }
# }

######################################


####### RNA COMPRESSED #######

# title = "Feature Importance Trajectories: RNA Compressed"
#
# dicts = {
#     "100": {
#         "cluster26_Compressed": 0.4568,
#         "cluster27_Compressed": 0.3769,
#         "cluster25_Compressed": 0.3440,
#         "cluster33_Compressed": 0.2875,
#         "cluster23_Compressed": 0.2504,
#         "cluster30_Compressed": 0.2173,
#         "cluster6_Compressed": 0.2056,
#         "cluster20_Compressed": 0.2010,
#         "cluster29_Compressed": 0.1933,
#         "cluster24_Compressed": 0.1893,
#         "cluster21_Compressed": 0.1851,
#         "cluster12_Compressed": 0.1758,
#         "cluster34_Compressed": 0.1738,
#         "cluster32_Compressed": 0.1674,
#         "cluster4_Compressed": 0.1657,
#         "cluster8_Compressed": 0.1399,
#         "cluster7_Compressed": 0.1210,
#         "cluster13_Compressed": 0.1154,
#         "cluster5_Compressed": 0.1145,
#         "cluster11_Compressed": 0.1071,
#         "cluster1_Compressed": 0.1046,
#         "cluster10_Compressed": 0.1035,
#         "cluster14_Compressed": 0.1030,
#         "cluster15_Compressed": 0.0986,
#         "cluster35_Compressed": 0.0971,
#         "cluster2_Compressed": 0.0944,
#         "cluster18_Compressed": 0.0878,
#         "cluster17_Compressed": 0.0859,
#         "cluster9_Compressed": 0.0834,
#         "cluster22_Compressed": 0.0822
#     },
#     "200": {
#         "cluster26_Compressed": 0.4490,
#         "cluster27_Compressed": 0.3968,
#         "cluster25_Compressed": 0.3377,
#         "cluster23_Compressed": 0.2593,
#         "cluster33_Compressed": 0.2562,
#         "cluster20_Compressed": 0.2247,
#         "cluster6_Compressed": 0.2108,
#         "cluster21_Compressed": 0.2002,
#         "cluster4_Compressed": 0.1979,
#         "cluster24_Compressed": 0.1879,
#         "cluster30_Compressed": 0.1769,
#         "cluster12_Compressed": 0.1720,
#         "cluster29_Compressed": 0.1583,
#         "cluster5_Compressed": 0.1565,
#         "cluster13_Compressed": 0.1465,
#         "cluster7_Compressed": 0.1464,
#         "cluster34_Compressed": 0.1426,
#         "cluster32_Compressed": 0.1375,
#         "cluster2_Compressed": 0.1339,
#         "cluster1_Compressed": 0.1248,
#         "cluster18_Compressed": 0.1115,
#         "cluster14_Compressed": 0.1051,
#         "cluster16_Compressed": 0.1019,
#         "cluster8_Compressed": 0.1013,
#         "cluster15_Compressed": 0.0995,
#         "cluster35_Compressed": 0.0978,
#         "cluster11_Compressed": 0.0870,
#         "cluster22_Compressed": 0.0868,
#         "cluster3_Compressed": 0.0845,
#         "cluster17_Compressed": 0.0842
#     },
#     "300": {
#         "cluster26_Compressed": 0.4490,
#         "cluster27_Compressed": 0.3989,
#         "cluster25_Compressed": 0.3538,
#         "cluster23_Compressed": 0.2727,
#         "cluster33_Compressed": 0.2613,
#         "cluster20_Compressed": 0.2343,
#         "cluster21_Compressed": 0.2148,
#         "cluster6_Compressed": 0.2114,
#         "cluster4_Compressed": 0.2101,
#         "cluster24_Compressed": 0.1984,
#         "cluster5_Compressed": 0.1786,
#         "cluster30_Compressed": 0.1781,
#         "cluster12_Compressed": 0.1669,
#         "cluster29_Compressed": 0.1630,
#         "cluster13_Compressed": 0.1558,
#         "cluster2_Compressed": 0.1442,
#         "cluster34_Compressed": 0.1401,
#         "cluster32_Compressed": 0.1401,
#         "cluster7_Compressed": 0.1341,
#         "cluster1_Compressed": 0.1242,
#         "cluster18_Compressed": 0.1182,
#         "cluster16_Compressed": 0.1134,
#         "cluster14_Compressed": 0.1084,
#         "cluster8_Compressed": 0.1013,
#         "cluster15_Compressed": 0.1010,
#         "cluster3_Compressed": 0.0950,
#         "cluster35_Compressed": 0.0942,
#         "cluster17_Compressed": 0.0908,
#         "cluster11_Compressed": 0.0871,
#         "cluster22_Compressed": 0.0869
#     },
#     "400": {
#         "cluster26_Compressed": 0.4423,
#         "cluster27_Compressed": 0.3987,
#         "cluster25_Compressed": 0.3565,
#         "cluster23_Compressed": 0.2781,
#         "cluster33_Compressed": 0.2577,
#         "cluster20_Compressed": 0.2273,
#         "cluster4_Compressed": 0.2154,
#         "cluster21_Compressed": 0.2074,
#         "cluster6_Compressed": 0.2053,
#         "cluster24_Compressed": 0.2011,
#         "cluster5_Compressed": 0.1880,
#         "cluster29_Compressed": 0.1864,
#         "cluster30_Compressed": 0.1727,
#         "cluster12_Compressed": 0.1651,
#         "cluster13_Compressed": 0.1568,
#         "cluster2_Compressed": 0.1561,
#         "cluster7_Compressed": 0.1446,
#         "cluster32_Compressed": 0.1404,
#         "cluster34_Compressed": 0.1357,
#         "cluster18_Compressed": 0.1284,
#         "cluster1_Compressed": 0.1275,
#         "cluster16_Compressed": 0.1227,
#         "cluster14_Compressed": 0.1073,
#         "cluster15_Compressed": 0.1051,
#         "cluster8_Compressed": 0.1020,
#         "cluster3_Compressed": 0.1007,
#         "cluster17_Compressed": 0.0984,
#         "cluster11_Compressed": 0.0941,
#         "cluster35_Compressed": 0.0905,
#         "cluster22_Compressed": 0.0869
#     },
#     "500": {
#         "cluster26_Compressed": 0.4415,
#         "cluster27_Compressed": 0.4080,
#         "cluster25_Compressed": 0.3522,
#         "cluster23_Compressed": 0.2807,
#         "cluster33_Compressed": 0.2655,
#         "cluster20_Compressed": 0.2239,
#         "cluster4_Compressed": 0.2080,
#         "cluster6_Compressed": 0.2053,
#         "cluster21_Compressed": 0.2026,
#         "cluster24_Compressed": 0.1951,
#         "cluster29_Compressed": 0.1853,
#         "cluster5_Compressed": 0.1843,
#         "cluster30_Compressed": 0.1723,
#         "cluster12_Compressed": 0.1681,
#         "cluster2_Compressed": 0.1572,
#         "cluster13_Compressed": 0.1565,
#         "cluster32_Compressed": 0.1541,
#         "cluster7_Compressed": 0.1451,
#         "cluster34_Compressed": 0.1361,
#         "cluster18_Compressed": 0.1315,
#         "cluster1_Compressed": 0.1259,
#         "cluster16_Compressed": 0.1247,
#         "cluster14_Compressed": 0.1146,
#         "cluster15_Compressed": 0.1070,
#         "cluster8_Compressed": 0.1030,
#         "cluster3_Compressed": 0.1022,
#         "cluster17_Compressed": 0.1016,
#         "cluster35_Compressed": 0.0947,
#         "cluster11_Compressed": 0.0934,
#         "cluster22_Compressed": 0.0906
#     }
# }

######################################

####### CYTOMETRY COMPRESSED #######

# title = "Feature Importance Trajectories: cytometry compressed"
#
# dicts = {
#     "100": {
#         "Cluster1_Compressed": 0.8030,
#         "Cluster2_Compressed": 0.6293,
#         "Cluster3_Compressed": 0.5239,
#         "Cluster4_Compressed": 0.5238,
#         "Cluster6_Compressed": 0.4050,
#         "Cluster5_Compressed": 0.2865
#     },
#     "200": {
#         "Cluster1_Compressed": 0.7732,
#         "Cluster2_Compressed": 0.5677,
#         "Cluster4_Compressed": 0.5348,
#         "Cluster3_Compressed": 0.5088,
#         "Cluster6_Compressed": 0.4222,
#         "Cluster5_Compressed": 0.3326
#     },
#     "300": {
#         "Cluster1_Compressed": 0.7766,
#         "Cluster2_Compressed": 0.5797,
#         "Cluster4_Compressed": 0.5486,
#         "Cluster3_Compressed": 0.5225,
#         "Cluster6_Compressed": 0.4315,
#         "Cluster5_Compressed": 0.3266
#     },
#     "400": {
#         "Cluster1_Compressed": 0.7688,
#         "Cluster2_Compressed": 0.5742,
#         "Cluster4_Compressed": 0.5342,
#         "Cluster3_Compressed": 0.5032,
#         "Cluster6_Compressed": 0.4195,
#         "Cluster5_Compressed": 0.3171
#     },
#     "500": {
#         "Cluster1_Compressed": 0.7630,
#         "Cluster2_Compressed": 0.5809,
#         "Cluster4_Compressed": 0.5321,
#         "Cluster3_Compressed": 0.4917,
#         "Cluster6_Compressed": 0.4170,
#         "Cluster5_Compressed": 0.3139
#     }
# }

######## CYTOKINES COMPRESSED ########

# title = "Feature Importance Trajectories: Cytokines Compressed"
#
# dicts = {
#     "100": {
#         "GRO_Compressed": 0.8564,
#         "HHV6.Status": 0.4319,
#         "IL-8_Compressed": 0.3999,
#         "Cluster4_Compressed": 0.1973,
#         "Cluster5_Compressed": 0.1797,
#         "Cluster3_Compressed": 0.1771,
#         "Cluster1_Compressed": 0.0644,
#         "Cluster7_Compressed": 0.0607,
#         "IP-10_Compressed": 0.0461,
#         "MCP-1_Compressed": 0.0395,
#         "Cluster2_Compressed": 0.0239,
#         "CMV.Status": 0.0000,
#         "EBV.Status": 0.0000,
#         "HSV1_2.Status": 0.0000,
#         "Cluster6_Compressed": 0.0000,
#         "Cluster8_Compressed": 0.0000,
#         "SCD40L_Compressed": 0.0000,
#         "IL12-p40_Compressed": 0.0000
#     },
#     "200": {
#         "GRO_Compressed": 0.8287,
#         "IL-8_Compressed": 0.3161,
#         "Cluster3_Compressed": 0.1735,
#         "Cluster4_Compressed": 0.1260,
#         "Cluster5_Compressed": 0.1228,
#         "HHV6.Status": 0.0751,
#         "Cluster1_Compressed": 0.0639,
#         "IP-10_Compressed": 0.0585,
#         "MCP-1_Compressed": 0.0363,
#         "Cluster2_Compressed": 0.0253,
#         "Cluster8_Compressed": 0.0052,
#         "CMV.Status": 0.0000,
#         "EBV.Status": 0.0000,
#         "HSV1_2.Status": 0.0000,
#         "Cluster6_Compressed": 0.0000,
#         "Cluster7_Compressed": 0.0000,
#         "SCD40L_Compressed": 0.0000,
#         "IL12-p40_Compressed": 0.0000
#     },
#     "300":{
#         "GRO_Compressed": 0.9565,
#         "IL-8_Compressed": 0.2809,
#         "Cluster3_Compressed": 0.1735,
#         "Cluster5_Compressed": 0.1141,
#         "Cluster4_Compressed": 0.0831,
#         "IP-10_Compressed": 0.0758,
#         "Cluster1_Compressed": 0.0456,
#         "MCP-1_Compressed": 0.0371,
#         "Cluster7_Compressed": 0.0216,
#         "Cluster2_Compressed": 0.0110,
#         "CMV.Status": 0.0000,
#         "EBV.Status": 0.0000,
#         "HSV1_2.Status": 0.0000,
#         "HHV6.Status": 0.0000,
#         "Cluster6_Compressed": 0.0000,
#         "Cluster8_Compressed": 0.0000,
#         "SCD40L_Compressed": 0.0000,
#         "IL12-p40_Compressed": 0.0000
#     },
#     "400": {
#         "GRO_Compressed": 1.0000,
#         "IL-8_Compressed": 0.2335,
#         "Cluster3_Compressed": 0.1735,
#         "Cluster5_Compressed": 0.1133,
#         "IP-10_Compressed": 0.0910,
#         "Cluster4_Compressed": 0.0727,
#         "Cluster1_Compressed": 0.0435,
#         "MCP-1_Compressed": 0.0412,
#         "Cluster2_Compressed": 0.0193,
#         "Cluster7_Compressed": 0.0163,
#         "CMV.Status": 0.0000,
#         "EBV.Status": 0.0000,
#         "HSV1_2.Status": 0.0000,
#         "HHV6.Status": 0.0000,
#         "Cluster6_Compressed": 0.0000,
#         "Cluster8_Compressed": 0.0000,
#         "SCD40L_Compressed": 0.0000,
#         "IL12-p40_Compressed": 0.0000
#     },
#     "500": {
#         "GRO_Compressed": 0.8340,
#         "IL-8_Compressed": 0.2354,
#         "Cluster3_Compressed": 0.1771,
#         "Cluster5_Compressed": 0.1156,
#         "HHV6.Status": 0.1103,
#         "IP-10_Compressed": 0.0990,
#         "MCP-1_Compressed": 0.0603,
#         "Cluster1_Compressed": 0.0415,
#         "Cluster4_Compressed": 0.0363,
#         "Cluster7_Compressed": 0.0216,
#         "Cluster2_Compressed": 0.0194,
#         "CMV.Status": 0.0000,
#         "EBV.Status": 0.0000,
#         "HSV1_2.Status": 0.0000,
#         "Cluster6_Compressed": 0.0000,
#         "Cluster8_Compressed": 0.0000,
#         "SCD40L_Compressed": 0.0000,
#         "IL12-p40_Compressed": 0.0000
#     }
# }

######################################

########## CLONAL DEPTH ############

# title = "Feature Importance Trajectories: Uncompressed Clonal depth"
#
# dicts = {
#     "100": {
#         "uniqueMoleculeFraction_ab": 1.0000,
#         "uniqueMoleculeFraction_gd": 0.1580
#     },
#     "200": {
#         "uniqueMoleculeFraction_ab": 1.0000,
#         "uniqueMoleculeFraction_gd": 0.1573
#     },
#     "300": {
#         "uniqueMoleculeFraction_ab": 1.0000,
#         "uniqueMoleculeFraction_gd": 0.1523
#     },
#     "400": {
#         "uniqueMoleculeFraction_ab": 1.0000,
#         "uniqueMoleculeFraction_gd": 0.1526
#     },
#     "500": {
#         "uniqueMoleculeFraction_ab": 1.0000,
#         "uniqueMoleculeFraction_gd": 0.1490
#     }
# }

######################################

########## CLONAL BREADTH ############

# title = "Feature Importance Trajectories: Uncompressed Clonal Breadth"
#
# dicts = {
#     "100": {
#         "fraction_sequences_ab": 1.0000,
#         "fraction_sequences_gd": 0.3304
#     },
#     "200": {
#         "fraction_sequences_ab": 1.0000,
#         "fraction_sequences_gd": 0.4875
#     },
#     "300": {
#         "fraction_sequences_ab": 1.0000,
#         "fraction_sequences_gd": 0.5089
#     },
#     "400": {
#         "fraction_sequences_ab": 1.0000,
#         "fraction_sequences_gd": 0.5089
#     },
#     "500": {
#         "fraction_sequences_ab": 1.0000,
#         "fraction_sequences_gd": 0.5161
#     }
# }

######################################

####### CYTOMETRY UNCOMPRESSED HEPATITIS B #######

title = "Feature Importance Trajectories: cytometry"
dicts = {
    "100": {
        "%GRA Day 0": 0.9674,
        "WBC Day 0": 0.6703,
        "%LYM Day 0": 0.6241,
        "RBC Day 0": 0.3324,
        "%MON Day 0": 0.2568,
        "PLT Day 0": 0.2324,
        "HGB Day 0": 0.2209,
        "HCT Day 0": 0.1826
    },
    "200": {
        "%GRA Day 0": 0.7928,
        "%LYM Day 0": 0.7182,
        "WBC Day 0": 0.5966,
        "%MON Day 0": 0.4885,
        "HCT Day 0": 0.3594,
        "RBC Day 0": 0.3410,
        "HGB Day 0": 0.2842,
        "PLT Day 0": 0.2567
    },
    "300":{
        "%GRA Day 0": 0.7773,
        "%LYM Day 0": 0.6304,
        "%MON Day 0": 0.5250,
        "WBC Day 0": 0.4656,
        "HCT Day 0": 0.3637,
        "RBC Day 0": 0.3559,
        "HGB Day 0": 0.2842,
        "PLT Day 0": 0.2704
    },
    "400" : {
        "%GRA Day 0": 0.7643,
        "%LYM Day 0": 0.5790,
        "%MON Day 0": 0.5655,
        "WBC Day 0": 0.4724,
        "RBC Day 0": 0.3586,
        "HCT Day 0": 0.2813,
        "HGB Day 0": 0.2808,
        "PLT Day 0": 0.2798
    },
    "500": {
        "%GRA Day 0": 0.7513,
        "%LYM Day 0": 0.6281,
        "%MON Day 0": 0.5130,
        "WBC Day 0": 0.4095,
        "RBC Day 0": 0.3803,
        "HCT Day 0": 0.3118,
        "PLT Day 0": 0.2892,
        "HGB Day 0": 0.2695
    },
    "600": {
        "%GRA Day 0": 0.7617,
        "%LYM Day 0": 0.6281,
        "%MON Day 0": 0.4939,
        "WBC Day 0": 0.4095,
        "RBC Day 0": 0.3613,
        "PLT Day 0": 0.2704,
        "HCT Day 0": 0.2678,
        "HGB Day 0": 0.1894
    },
    "700": {
        "%GRA Day 0": 0.6785,
        "%LYM Day 0": 0.5295,
        "%MON Day 0": 0.4446,
        "WBC Day 0": 0.4095,
        "RBC Day 0": 0.3803,
        "HCT Day 0": 0.2940,
        "PLT Day 0": 0.2546,
        "HGB Day 0": 0.1925
    },
    "800": {
        "%GRA Day 0": 0.6705,
        "%LYM Day 0": 0.5281,
        "%MON Day 0": 0.4407,
        "WBC Day 0": 0.4376,
        "RBC Day 0": 0.3803,
        "HCT Day 0": 0.3005,
        "PLT Day 0": 0.2625,
        "HGB Day 0": 0.2121
    },
    "900": {
        "%GRA Day 0": 0.6785,
        "%LYM Day 0": 0.5486,
        "WBC Day 0": 0.4824,
        "%MON Day 0": 0.4275,
        "RBC Day 0": 0.3803,
        "PLT Day 0": 0.3145,
        "HCT Day 0": 0.2588,
        "HGB Day 0": 0.2153
    },
    "1000": {
        "%GRA Day 0": 0.7386,
        "%LYM Day 0": 0.5997,
        "WBC Day 0": 0.4824,
        "%MON Day 0": 0.4275,
        "RBC Day 0": 0.4076,
        "PLT Day 0": 0.3145,
        "HCT Day 0": 0.2678,
        "HGB Day 0": 0.2317
    },
    "1100": {
        "%GRA Day 0": 0.7384,
        "%LYM Day 0": 0.6106,
        "WBC Day 0": 0.5214,
        "%MON Day 0": 0.4368,
        "RBC Day 0": 0.3803,
        "PLT Day 0": 0.3145,
        "HCT Day 0": 0.2678,
        "HGB Day 0": 0.2317
    },
    "1200": {
        "%GRA Day 0": 0.7513,
        "%LYM Day 0": 0.6106,
        "WBC Day 0": 0.5214,
        "%MON Day 0": 0.3807,
        "RBC Day 0": 0.3803,
        "PLT Day 0": 0.3145,
        "HCT Day 0": 0.2732,
        "HGB Day 0": 0.2695
    },
    "1300":  {
        "%GRA Day 0": 0.7513,
        "%LYM Day 0": 0.6284,
        "WBC Day 0": 0.5214,
        "RBC Day 0": 0.3803,
        "%MON Day 0": 0.3739,
        "PLT Day 0": 0.2892,
        "HGB Day 0": 0.2774,
        "HCT Day 0": 0.2678
    },
    "1400": {
        "%GRA Day 0": 0.7449,
        "%LYM Day 0": 0.6283,
        "WBC Day 0": 0.5711,
        "RBC Day 0": 0.3785,
        "%MON Day 0": 0.3773,
        "PLT Day 0": 0.3145,
        "HGB Day 0": 0.2808,
        "HCT Day 0": 0.2633
    },
    "1500": {
        "%GRA Day 0": 0.7384,
        "%LYM Day 0": 0.6281,
        "WBC Day 0": 0.5203,
        "RBC Day 0": 0.4207,
        "%MON Day 0": 0.4061,
        "PLT Day 0": 0.3145,
        "HGB Day 0": 0.2774,
        "HCT Day 0": 0.2678
    },
    "1600": {
        "%GRA Day 0": 0.7385,
        "%LYM Day 0": 0.6283,
        "WBC Day 0": 0.5463,
        "RBC Day 0": 0.4141,
        "%MON Day 0": 0.3938,
        "PLT Day 0": 0.3175,
        "HGB Day 0": 0.2735,
        "HCT Day 0": 0.2633
    },
    "1700": {
        "%GRA Day 0": 0.6978,
        "%LYM Day 0": 0.6281,
        "WBC Day 0": 0.5214,
        "%MON Day 0": 0.4061,
        "RBC Day 0": 0.3803,
        "PLT Day 0": 0.3145,
        "HGB Day 0": 0.2695,
        "HCT Day 0": 0.2678
    },
    "1800": {
        "%GRA Day 0": 0.6978,
        "%LYM Day 0": 0.6244,
        "WBC Day 0": 0.5214,
        "%MON Day 0": 0.4275,
        "RBC Day 0": 0.3803,
        "PLT Day 0": 0.3145,
        "HGB Day 0": 0.2609,
        "HCT Day 0": 0.2509
    },
    "1900": {
        "%GRA Day 0": 0.6978,
        "%LYM Day 0": 0.6244,
        "WBC Day 0": 0.5214,
        "%MON Day 0": 0.4275,
        "RBC Day 0": 0.3803,
        "PLT Day 0": 0.3145,
        "HGB Day 0": 0.2609,
        "HCT Day 0": 0.2509
    },
    "2000": {
        "%GRA Day 0": 0.6468,
        "%LYM Day 0": 0.5874,
        "WBC Day 0": 0.5013,
        "%MON Day 0": 0.4106,
        "RBC Day 0": 0.4102,
        "PLT Day 0": 0.3055,
        "HCT Day 0": 0.2517,
        "HGB Day 0": 0.2177
    },
    "2100": {
        "%GRA Day 0": 0.6785,
        "%LYM Day 0": 0.5782,
        "WBC Day 0": 0.5203,
        "RBC Day 0": 0.4165,
        "%MON Day 0": 0.4137,
        "PLT Day 0": 0.2892,
        "HCT Day 0": 0.2526,
        "HGB Day 0": 0.2201
    },
    "2200": {
        "%GRA Day 0": 0.6625,
        "%LYM Day 0": 0.5782,
        "WBC Day 0": 0.5214,
        "%MON Day 0": 0.4275,
        "RBC Day 0": 0.4207,
        "PLT Day 0": 0.2939,
        "HCT Day 0": 0.2526,
        "HGB Day 0": 0.2317
    },
    "2300": {
        "%GRA Day 0": 0.6644,
        "%LYM Day 0": 0.5486,
        "WBC Day 0": 0.5214,
        "%MON Day 0": 0.4446,
        "RBC Day 0": 0.4361,
        "PLT Day 0": 0.2965,
        "HCT Day 0": 0.2526,
        "HGB Day 0": 0.2201
    },
    "2400": {
        "%GRA Day 0": 0.6644,
        "%LYM Day 0": 0.5750,
        "WBC Day 0": 0.5711,
        "%MON Day 0": 0.4368,
        "RBC Day 0": 0.4245,
        "PLT Day 0": 0.2965,
        "HCT Day 0": 0.2578,
        "HGB Day 0": 0.2317
    },
    "2500": {
        "%GRA Day 0": 0.6908,
        "%LYM Day 0": 0.5782,
        "WBC Day 0": 0.5214,
        "%MON Day 0": 0.4446,
        "RBC Day 0": 0.4245,
        "PLT Day 0": 0.2892,
        "HCT Day 0": 0.2578,
        "HGB Day 0": 0.2248
    },
    "2600": {
        "%GRA Day 0": 0.6978,
        "%LYM Day 0": 0.6106,
        "WBC Day 0": 0.5203,
        "%MON Day 0": 0.4577,
        "RBC Day 0": 0.4245,
        "PLT Day 0": 0.2796,
        "HCT Day 0": 0.2578,
        "HGB Day 0": 0.2248
    },
    "2700" : {
        "%GRA Day 0": 0.6908,
        "%LYM Day 0": 0.5997,
        "WBC Day 0": 0.5203,
        "%MON Day 0": 0.4577,
        "RBC Day 0": 0.4207,
        "PLT Day 0": 0.2939,
        "HCT Day 0": 0.2578,
        "HGB Day 0": 0.2317
    },
    "2800" : {
        "%GRA Day 0": 0.6978,
        "%LYM Day 0": 0.6018,
        "WBC Day 0": 0.5209,
        "%MON Day 0": 0.4322,
        "RBC Day 0": 0.4241,
        "PLT Day 0": 0.2844,
        "HCT Day 0": 0.2656,
        "HGB Day 0": 0.2445
    },
    "2900": {
        "%GRA Day 0": 0.6908,
        "%LYM Day 0": 0.5766,
        "WBC Day 0": 0.5125,
        "%MON Day 0": 0.4511,
        "RBC Day 0": 0.4241,
        "PLT Day 0": 0.2916,
        "HCT Day 0": 0.2816,
        "HGB Day 0": 0.2646
    },
    "3000": {
        "%GRA Day 0": 0.6908,
        "%LYM Day 0": 0.5782,
        "WBC Day 0": 0.5047,
        "%MON Day 0": 0.4577,
        "RBC Day 0": 0.4238,
        "PLT Day 0": 0.2892,
        "HCT Day 0": 0.2825,
        "HGB Day 0": 0.2682
    },
    "3100": {
        "%GRA Day 0": 0.6908,
        "%LYM Day 0": 0.5997,
        "WBC Day 0": 0.5047,
        "%MON Day 0": 0.4446,
        "RBC Day 0": 0.4361,
        "HCT Day 0": 0.2893,
        "PLT Day 0": 0.2892,
        "HGB Day 0": 0.2685
    },
    "3200": {
        "%GRA Day 0": 0.6908,
        "%LYM Day 0": 0.5958,
        "WBC Day 0": 0.5047,
        "RBC Day 0": 0.4361,
        "%MON Day 0": 0.4275,
        "PLT Day 0": 0.2892,
        "HCT Day 0": 0.2807,
        "HGB Day 0": 0.2682
    },
    "3300": {
        "%GRA Day 0": 0.6908,
        "%LYM Day 0": 0.5978,
        "WBC Day 0": 0.5000,
        "%MON Day 0": 0.4399,
        "RBC Day 0": 0.4370,
        "PLT Day 0": 0.2916,
        "HCT Day 0": 0.2859,
        "HGB Day 0": 0.2682
    },
    "3400" : {
        "%GRA Day 0": 0.6978,
        "%LYM Day 0": 0.5870,
        "WBC Day 0": 0.4909,
        "%MON Day 0": 0.4596,
        "RBC Day 0": 0.4370,
        "HCT Day 0": 0.2859,
        "PLT Day 0": 0.2795,
        "HGB Day 0": 0.2682
    },
    "3500": {
        "%GRA Day 0": 0.6714,
        "%LYM Day 0": 0.5570,
        "WBC Day 0": 0.4818,
        "%MON Day 0": 0.4689,
        "RBC Day 0": 0.4241,
        "HCT Day 0": 0.2928,
        "PLT Day 0": 0.2818,
        "HGB Day 0": 0.2682
    },
    "3600": {
        "%GRA Day 0": 0.6634,
        "%LYM Day 0": 0.5541,
        "WBC Day 0": 0.4651,
        "%MON Day 0": 0.4641,
        "RBC Day 0": 0.4188,
        "HCT Day 0": 0.2950,
        "PLT Day 0": 0.2806,
        "HGB Day 0": 0.2684
    },
    "3700": {
        "%GRA Day 0": 0.6531,
        "%LYM Day 0": 0.5486,
        "%MON Day 0": 0.4711,
        "WBC Day 0": 0.4646,
        "RBC Day 0": 0.4159,
        "HCT Day 0": 0.2972,
        "PLT Day 0": 0.2820,
        "HGB Day 0": 0.2695
    },
    "3800": {
        "%GRA Day 0": 0.6571,
        "%LYM Day 0": 0.5486,
        "%MON Day 0": 0.4711,
        "WBC Day 0": 0.4646,
        "RBC Day 0": 0.4128,
        "HCT Day 0": 0.2972,
        "PLT Day 0": 0.2820,
        "HGB Day 0": 0.2695
    },
    "3900": {
        "%GRA Day 0": 0.6585,
        "%LYM Day 0": 0.5489,
        "%MON Day 0": 0.4793,
        "WBC Day 0": 0.4651,
        "RBC Day 0": 0.4059,
        "HCT Day 0": 0.3001,
        "PLT Day 0": 0.2894,
        "HGB Day 0": 0.2690
    },
    "4000": {
        "%GRA Day 0": 0.6551,
        "%LYM Day 0": 0.5486,
        "%MON Day 0": 0.4843,
        "WBC Day 0": 0.4651,
        "RBC Day 0": 0.4059,
        "HCT Day 0": 0.3001,
        "PLT Day 0": 0.2919,
        "HGB Day 0": 0.2690
    },
    "4100":  {
        "%GRA Day 0": 0.6684,
        "%LYM Day 0": 0.5725,
        "%MON Day 0": 0.4748,
        "WBC Day 0": 0.4683,
        "RBC Day 0": 0.4059,
        "HCT Day 0": 0.2987,
        "PLT Day 0": 0.2897,
        "HGB Day 0": 0.2684
    },
    "4200": {
        "%GRA Day 0": 0.6724,
        "%LYM Day 0": 0.5750,
        "%MON Day 0": 0.4817,
        "WBC Day 0": 0.4656,
        "RBC Day 0": 0.4159,
        "HCT Day 0": 0.2972,
        "PLT Day 0": 0.2897,
        "HGB Day 0": 0.2682
    },
    "4300": {
        "%GRA Day 0": 0.6684,
        "%LYM Day 0": 0.5725,
        "%MON Day 0": 0.4843,
        "WBC Day 0": 0.4651,
        "RBC Day 0": 0.4162,
        "HCT Day 0": 0.3001,
        "PLT Day 0": 0.2894,
        "HGB Day 0": 0.2670
    },
    "4400": {
        "%GRA Day 0": 0.6634,
        "%LYM Day 0": 0.5646,
        "WBC Day 0": 0.4596,
        "%MON Day 0": 0.4793,
        "RBC Day 0": 0.4162,
        "HCT Day 0": 0.3001,
        "PLT Day 0": 0.2818,
        "HGB Day 0": 0.2654
    },
    "4500": {
        "%GRA Day 0": 0.6599,
        "%LYM Day 0": 0.5750,
        "%MON Day 0": 0.4817,
        "WBC Day 0": 0.4596,
        "RBC Day 0": 0.4165,
        "HCT Day 0": 0.3001,
        "PLT Day 0": 0.2815,
        "HGB Day 0": 0.2609
    },
    "4600": {
        "%GRA Day 0": 0.6612,
        "%LYM Day 0": 0.5725,
        "%MON Day 0": 0.4724,
        "WBC Day 0": 0.4400,
        "RBC Day 0": 0.4143,
        "HCT Day 0": 0.3001,
        "PLT Day 0": 0.2856,
        "HGB Day 0": 0.2630
    },
    "4700": {
        "%GRA Day 0": 0.6684,
        "%LYM Day 0": 0.5766,
        "WBC Day 0": 0.4320,
        "%MON Day 0": 0.4728,
        "RBC Day 0": 0.4162,
        "HCT Day 0": 0.2966,
        "PLT Day 0": 0.2894,
        "HGB Day 0": 0.2517
    },
    "4800": {
        "%GRA Day 0": 0.6585,
        "%LYM Day 0": 0.5646,
        "%MON Day 0": 0.4728,
        "WBC Day 0": 0.4320,
        "RBC Day 0": 0.4102,
        "HCT Day 0": 0.2988,
        "PLT Day 0": 0.2919,
        "HGB Day 0": 0.2630
    },
    "4900": {
        "%GRA Day 0": 0.6585,
        "%LYM Day 0": 0.5718,
        "%MON Day 0": 0.4728,
        "WBC Day 0": 0.4286,
        "RBC Day 0": 0.4091,
        "HCT Day 0": 0.3015,
        "PLT Day 0": 0.2942,
        "HGB Day 0": 0.2532
    },
    "5000": {
        "%GRA Day 0": 0.6634,
        "%LYM Day 0": 0.5646,
        "%MON Day 0": 0.4724,
        "WBC Day 0": 0.4263,
        "RBC Day 0": 0.4059,
        "HCT Day 0": 0.3033,
        "PLT Day 0": 0.2942,
        "HGB Day 0": 0.2630
    }




}

##################################################

# Function to prepare data for plotting
def prepare_plot_data(iteration_data_dict):
    # Extract and sort iteration numbers
    iterations = sorted([int(k) for k in iteration_data_dict.keys()])

    # Get all unique features across all iterations
    all_features = sorted(list(set.union(*[set(iteration_data_dict[str(i)].keys()) for i in iterations])))

    feature_trajectories = {feature: [] for feature in all_features}

    for i in iterations:
        current_dict = iteration_data_dict[str(i)]
        for feature in all_features:
            # Get value, default to 0.0 if not present in current_dict
            feature_trajectories[feature].append(current_dict.get(feature, 0.0))

    return feature_trajectories, iterations


def plot_feature_trajectories(feature_trajectories, iterations, title):
    plt.figure(figsize=(12, 8))

    # Filter out features that are consistently zero across all iterations
    non_zero_features = {
        feature: trajectory for feature, trajectory in feature_trajectories.items()
        if any(val > 0 for val in trajectory)
    }

    for feature, trajectory in non_zero_features.items():
        plt.plot(iterations, trajectory, marker='o', linestyle='-', label=feature)

    plt.title(title)
    plt.xlabel("Number of Iterations (Splits)")
    plt.ylabel("Median of Mean SHAP Values")
    plt.grid(True, linestyle='--', alpha=0.7)
    step = max(1, len(iterations) // 21)
    plt.xticks(iterations[::step], rotation=45, ha='right')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

cytokines_uncompressed_trajectories, iterations = prepare_plot_data(dicts)
plot_feature_trajectories(
    cytokines_uncompressed_trajectories,
    iterations,
    title,
)