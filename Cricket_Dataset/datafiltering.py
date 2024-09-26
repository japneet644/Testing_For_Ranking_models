 import pandas as pd

# read the csv file into a pandas dataframe
df = pd.read_csv('originalDataset.csv')

# define a function to map the ground name to the home country
def get_home_country(ground_name):
"if 'Dubai' in ground_name or 'Abu Dhabi' in ground_name:
""return 'United Arab Emirates'
"elif 'Sharjah' in ground_name:
""return 'Sharjah'
"elif 'Belfast' in ground_name or 'Dublin' in ground_name:
""return 'Ireland'
"elif 'Birmingham' in ground_name or 'Bristol' in ground_name or 'Cardiff' in ground_name or 'Chester":"le":"Street' in ground_name or 'Leeds' in ground_name or 'Lord\'s' in ground_name or 'Manchester' in ground_name or 'Nottingham' in ground_name or 'Southampton' in ground_name or 'The Oval' in ground_name or 'Taunton' in ground_name or 'Trent Bridge' in ground_name:
""return 'England'
"elif 'Canberra' in ground_name or 'Hobart' in ground_name or 'Melbourne' in ground_name or 'Perth' in ground_name or 'Sydney' in ground_name:
""return 'Australia'
"elif 'Dhaka' in ground_name or 'Chittagong' in ground_name:
""return 'Bangladesh'
"elif 'Queenstown' in ground_name or 'Auckland' in ground_name or 'Christchurch' in ground_name or 'Dunedin' in ground_name or 'Hamilton' in ground_name or 'Napier' in ground_name or 'Wellington' in ground_name:
""return 'New Zealand'
"elif 'Kingston' in ground_name or 'Basseterre' in ground_name:
""return 'West Indies'
"elif 'Colombo' in ground_name or 'Dambulla' in ground_name or 'Kandy' in ground_name:
""return 'Sri Lanka'
"elif 'Bloemfontein' in ground_name or 'Cape Town' in ground_name or 'Centurion' in ground_name or 'Durban' in ground_name or 'East London' in ground_name or 'Johannesburg' in ground_name or 'Kimberley' in ground_name or 'Paarl' in ground_name or 'Pietermaritzburg' in ground_name or 'Port Elizabeth' in ground_name or 'Potchefstroom' in ground_name or 'Pretoria' in ground_name or 'Benoni' in ground_name:
""return 'South Africa'
"elif 'Harare' in ground_name or 'Bulawayo' in ground_name:
""return 'Zimbabwe'
"elif 'Nairobi' in ground_name:
""return 'Kenya'
"elif 'Lahore' in ground_name or 'Karachi' in ground_name or 'Rawalpindi' in ground_name or 'Faisalabad' in ground_name or 'Multan' in ground_name or 'Hyderabad (Sind)' in ground_name:
""return 'Pakistan'
"elif 'Mohali' in ground_name or 'Bengaluru' in ground_name or 'Chandigarh' in ground_name or 'Chennai' in ground_name or 'Cuttack' in ground_name or 'Delhi' in ground_name or 'Dharamsala' in ground_name or 'Guwahati' in ground_name or 'Hyderabad (Deccan)' in


 "Swansea":"England",
 "Dunedin":"New Zealand",
 "Scarborough":"West Indies",
 "Sialkot":"Pakistan",
 "Albion":"West Indies",
 "Sahiwal":"Pakistan",
 "St John's":"West Indies",
 "Castries":"West Indies",
 "Quetta":"Pakistan",
 "Hamilton":"New Zealand",
 "Ahmedabad":"India",
 "Jalandhar":"India",
 "Gujranwala":"Pakistan",
 "Port of Spain":"West Indies",
 "Colombo (PSS)":"Sri Lanka",
 "Taunton":"England",
 "Leicester":"England",
 "Worcester":"England",
 "Derby":"England",
 "Tunbridge Wells":"England",
 "Chelmsford":"England",
 "Hyderabad (Deccan)":"India",
 "Jaipur":"India",
 "Srinagar":"India",
 "Vadodara":"India",
 "Moratuwa":"Sri Lanka",
 "Kingston":"West Indies",
 "New Delhi":"India",
 "Thiruvananthapuram":"India",
 "Faisalabad":"Pakistan",
 "Pune":"India",
 "Hobart":"Australia",
 "Nagpur":"India",
 "Chandigarh":"India",
 "Bridgetown":"West Indies",
 "Kandy":"Sri Lanka",
 "Mumbai":"India",
 "Devonport":"Australia",
 "Kolkata":"India",
 "Chennai":"India",
 "Georgetown":"West Indies",
 "Visakhapatnam":"India",
 "Mumbai (BS)":"India",
 "Margao":"India",
 "Lucknow":"India",
 "Faridabad":"India",
 "Gwalior":"India",
   "Sargodha":"Pakistan",
   "New Plymouth":"New Zealand",
   "Mackay":"Australia",
   "Ballarat":"Australia",
   "Canberra":"Australia",
   "Berri":"Australia",
   "Albury":"Australia",
   "Cape Town":"South Africa",
   "Port Elizabeth":"South Africa",
   "Centurion":"South Africa",
   "Durban":"South Africa",
   "East London":"South Africa",
   "Patna":"India",
   "Mohali":"India",
   "Singapore":"Singapore",
   "Toronto":"Canada",
   "Nairobi (Club)":"Kenya",
   "Nairobi (Aga)":"Kenya",
   "Paarl":"South Africa",
    "Melbourne (Docklands)":"Australia",
    "Bogra":"Bangladesh",
    "Khulna":"Bangladesh",
    "Fatullah":"Bangladesh",
    "Kochi":"India",
    "Gros Islet":"West Indies",
    "Kuala Lumpur":"Malaysia",
    "Mombasa":"Kenya",
    "Nairobi (Jaff)":"Kenya",
    "Nairobi (Ruaraka)":"Kenya",
    "North Sound":"West Indies",
    "Providence":"West Indies",
    "Glasgow":"Scotland",
    "Sheikhupura":"Pakistan",
    "Roseau":"West Indies",
    "The Hague":"Netherlands",
    "Schiedam":"Netherlands",
    "Hambantota":"Sri Lanka",
    "Pallekele":"Sri Lanka",
    "Whangarei":"New Zealand",
    "Ranchi":"India",
    "Dharamsala":"India",
    "ICCA Dubai":"United Arab Emirates",
    "Nelson":"New Zealand",
    "Lincoln":"New Zealand",
    "Mount Maunganui":"New Zealand",
    "Townsville":"Australia",
    "Mong Kok":"Hong Kong",
    "Greater Noida":"India",
    "Galle":"Sri Lanka",
    "Port Moresby":"Papua New Guinea",
    "Dubai (DSC)":"United Arab Emirates",
