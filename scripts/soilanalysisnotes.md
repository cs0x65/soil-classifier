pH for fertile soil: range 6-7/6.2 - 7.3
queries that have ph:
 - in optimal range
 - > optimal range i.e. alkaline
 - < optimal range i.e. acidic
 ==

 clustering based on values/ranges of macro and micro nutrients
 ==

 soils having optimal, less and higher that normal macro nutrients
 queries for 3 ranges of macro nutrients values: optimum, low, high

 soils having optimal, less and higher that normal micro nutrients
 queries for 3 ranges of micro nutrients values: optimum, low, high
 ==

macro nutrients from air & water: carbon, oxygen and hydrogen
macro nutrients : av_p, av_k, av_s
micro nutrients : av_zn, av_b, av_fe, av_cu, av_mn
==

pH:
===
General pH for fertile soil: range 6-7/6.2 - 7.3
    acidic < 6
    alkaline > 7.3
    6 <= optimal <= 7.3

vegetables and row crops: 5.8 <= optimal <= 6.5
roses, turfgrasses, fruits and nuts: 5.5 <= optimal <= 5.8
crops/plants suffer when ph < 4.8

EC:
===
0.5 dS/m deci-siemens/meter > cause salt injury 
indicates salinity
normal <= 0.5 <= marginally high <= 1.5 <= excessive(salt injury)

Ranges of the nutrients:
===


Conversions based on FSA2118 paper:

Row crops and forages
--
macro nutrients:
av_p (kg/ha): very low < 35.8672, low 35.8672 - 56.0426, medium 58.2843 - 78.4596, optimum 80.7013 - 112.085, above optimum 112.085 > 
av_k (kg/ha): very low < 136.744, low 136.744 - 201.753, medium 203.995 - 291.421, optimum 293.663 - 392.298, above optimum 392.298 >
av_s (ppm): optimum <=10 
==
micro nutrients:
av_zn (ppm):  very low < 1.6, low 1.6 - 3.0, medium 3.1 - 4.0, optimum 4.0 - 8.0, above optimum 8.0 >
av_b (ppm): 
av_fe (ppm):
av_cu (ppm): optimal < 1.0
av_mn (ppm): optimal < 40

Vegetables
--
av_p (kg/ha): optimal >= 168.128
av_k (kg/ha): same as Row crops and forages
very low < 136.744, low 136.744 - 201.753, medium 203.995 - 291.421, optimum 293.663 - 392.298, above optimum 392.298 >

Fruits
--
av_p (kg/ha): optimal >= 56.0426
av_k (kg/ha): optimal >= 201.753



===
The soil nutrients specifically targeted for a crop, so can be used for crop suitability recommendation as well. 

Specific values range: 
1. Soil fertility norms for Sathgudi sweet orange/Citrus sinensis
    Table 2: Soil fertility norms (derived from DRIS based analysis) developed for Sathgudi sweet orange orchards of Andhra Pradesh
2. DRIS Norms and their Field Validation in Nagpur Mandarin:
    Table 4: Soil analysis based fertility indices in relation to fruit yield of mandarin orchards
3. IJAER Guntur Crop Specific Suitability v12n23_123
    Table 2. Suitable ranges for Jowar
 

   
Classifications implementation
--
Single label classification:

pH:
===
Label header: ph_class 
general pH for fertile soil: acidic < 6,alkaline > 7.3, 6 <= optimal <= 7.3
>> Done

EC:
===
Label header: ec_class
Acceptable EC for healthy soil:  normal <= 0.5 dS/m deci-siemens/meter < cause salt injury
indicates salinity
normal <= 0.5 <= marginally high <= 1.5 <= excessive(salt injury)
>> TBD


Multi label classification:

pH:
===
Label header: ph_veg_and_row_crops 
optimal pH range for soil to be fertile for veg & row crops: 5.8 <= optimal <= 6.5
>> Done

Label header: ph_fruits_and_nuts 
optimal pH range for soil to be fertile for fruits & nuts: 5.5 <= optimal <= 5.8
>> Done