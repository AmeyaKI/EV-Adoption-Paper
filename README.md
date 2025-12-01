# EV-Adoption-Paper
EV Research Models and Datasets

# Dataset Characteristics

Timeframe: 2014-2022

States: 46 states. States omitted were Alaska, Colorado, Hawaii, and New Hampshire. DC was also omitted. Omissions due to lack of data points for certain variables.


# Dataset Sources


**Infrastructure**: https://afdc.energy.gov/stations/states 

  Units: number of charging outlets 
  EVSE - EV Supply Equipment

**Electricity Price**: https://www.eia.gov/electricity/monthly/

  Annual September datasets used.
  Units: Cents per Kilowatt. 
  Based on Residential electricity prices.
  
**Gas Price**: https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMA_EPM0_PWG_SAL_DPG&f=M

  Total Gasoline Wholesale/Resale Price by Refiners
  
  Units: Dollars per Gallon
  
  Note: The Link attached is specifically for Alabama. Each state has a different link. To find the original data for each individual state, search up "[State Name] Total Gasoline Wholesale/Resale Price by Refiners"
  
  While data did exist on Retail Gasoline Prices /  prices closer to those seen at gas stations, unfortunately this data was either only limited to the US writ large (no data on individual states) or didn't exist for a sufficient number of states (ie. only 6-8 states published retail gasoline prices). 
  
  There were some approximations made to include more states. Specifically, Data for Vermont gas prices in the years 2018, 2019, 2021, and 2022 were calculated using data from the other available yaers of 2014, 2015, 2016, 2017, 2020 and approximations of average annual increase/decrease across neighboring states of NY, MA, NH, ME, RI, CT. 


**Median Income**: https://fred.stlouisfed.org/release/tables?rid=249&eid=259515&od=# 

  Median annual income in dollars per household

**Political Control**: https://ballotpedia.org/Alabama_State_Legislature

  Scale from 0 to 1 where 0 is 100% Republican controlled and 1 is 100% Democrat controlled
  
  Note: As was the case for gas price, the link attached is specifically for Alabama as every state has a different link. TO find the original data about state legislatures for each individual state, serach up "[State Name] State Legislature" on Ballotpedia.

**Car Price Difference**: https://mediaroom.kbb.com/new-car-transaction-prices-up-september-2015-volkswagen-down

  Data came from Kelley Blue Book which is the only available source for calculations regarding average price of traditional, gas-powered, ICE vehicles (ICEVs) and electric vehicles (EVs) over a long enough period of time.
  
  Annual data points from Kelley Blue Book's September data reports were used. The selection of using September was purely random, yet uniformly consistent across the timeframe of teh data examined (2014-2022), as was also the case for electricity price.

**Lithium Price**: https://www.energy.gov/eere/vehicles/articles/fotw-1272-january-9-2023-electric-vehicle-battery-pack-costs-2022-are-nearly

  Data from the US Department of Energy's Vehicle Technologies Office. 
  Units: Dollars per kWH

**EV Range**: https://www.energy.gov/eere/vehicles/articles/fotw-1323-january-1-2024-top-range-model-year-2023-evs-was-516-miles-single

  Data from the US Department of Energy citing the EPA.
  Units: Miles.

**Combined EV Tax Credit**:

  Federal Credit: https://afdc.energy.gov/laws/409
  
  State Credits: https://taxfoundation.org/data/all/state/electric-vehicles-ev-taxes-state/

  Combined EV Tax Credit consists of federal credit plus state credit.
  Federal Credit is $7,500.
  State Credits vary depending on the state.

**Motor Fuel Tax**: https://www.fhwa.dot.gov/policyinformation/statistics/2022/ 

  Table 8.2.3: https://www.fhwa.dot.gov/policyinformation/statistics/2022/mf205.cfm
  
  Units: Cents per Gallon

**HOV Access**: https://afdc.energy.gov/laws/HOV 

  
