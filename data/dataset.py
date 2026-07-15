import pandas as pd
import numpy as np
import random

# ------------------ Setup ------------------
np.random.seed(42)
random.seed(42)

n_rows = 1000

# ------------------ Helper Lists ------------------
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weather_options = ['Normal','Rainy','Hot','Cloudy','Overcast']
cuisines = ['Cafe','FastFood','Bakery','SouthIndian','Italian','Chinese']
traffic_options = ['Low','Medium','High']
parking_options = ['Available','Limited','Full']
cloudiness_options = ['Sunny','Partly Cloudy','Overcast','Cloudy']
attraction_options = ['Mall','Office','School','Park','Cinema','None']
seasons = ['Summer','Monsoon','Winter']
noise_map = {'Low':1,'Medium':2,'High':3}

# ------------------ Generate Columns ------------------
day_col = [random.choice(days) for _ in range(n_rows)]
hour_col = [random.randint(8,22) for _ in range(n_rows)]
weekend_col = [1 if day in ['Saturday','Sunday'] else 0 for day in day_col]

weather_col = [random.choices(weather_options, weights=[0.5,0.2,0.15,0.1,0.05])[0] for _ in range(n_rows)]
temperature_col = [round(random.uniform(20,35),1) for _ in range(n_rows)]
humidity_col = [round(random.uniform(30,90),1) for _ in range(n_rows)]
cloudiness_col = [random.choice(cloudiness_options) for _ in range(n_rows)]

event_col = [random.choices([0,1], weights=[0.8,0.2])[0] for _ in range(n_rows)]
holiday_col = [random.choices([0,1], weights=[0.85,0.15])[0] for _ in range(n_rows)]
festival_col = [random.choices([0,1], weights=[0.9,0.1])[0] for _ in range(n_rows)]
offer_col = [random.choices([0,1], weights=[0.7,0.3])[0] for _ in range(n_rows)]
season_col = [random.choice(seasons) for _ in range(n_rows)]

ratings_col = [round(random.uniform(3.5,5.0),1) for _ in range(n_rows)]
cuisine_col = [random.choice(cuisines) for _ in range(n_rows)]

traffic_col = [random.choices(traffic_options, weights=[0.5,0.35,0.15])[0] for _ in range(n_rows)]
parking_col = [random.choices(parking_options, weights=[0.6,0.3,0.1])[0] for _ in range(n_rows)]
parking_distance_col = [round(random.uniform(10,200),1) for _ in range(n_rows)]

attractions_col = [random.choice(attraction_options) for _ in range(n_rows)]
aqi_col = [random.randint(50,200) for _ in range(n_rows)]

user_max_crowd_col = [random.randint(10,60) for _ in range(n_rows)]
user_noise_tolerance_col = [random.choice(list(noise_map.keys())) for _ in range(n_rows)]
user_preferred_temp_col = [round(random.uniform(22,30),1) for _ in range(n_rows)]
user_preferred_humidity_col = [round(random.uniform(30,70),1) for _ in range(n_rows)]
user_accessibility_col = [random.choice(['Yes','No']) for _ in range(n_rows)]
user_seating_col = [random.choice(['Indoor','Outdoor','Window','Booth']) for _ in range(n_rows)]

# ------------------ Footfall ------------------
footfall_col = []
for i in range(n_rows):
    base = 20 + 5*weekend_col[i] + 5*event_col[i] + 5*holiday_col[i] + 5*festival_col[i]
    if weather_col[i] in ['Rainy','Overcast']:
        base += 5
    base += int((hour_col[i]-8)*1.5)
    base += random.randint(-5,5)
    footfall_col.append(max(5, base))

# ------------------ Crowd Density ------------------
crowdtrend_col = []
for f in footfall_col:
    if f < 20:
        crowdtrend_col.append('Low')
    elif f < 40:
        crowdtrend_col.append('Medium')
    elif f < 60:
        crowdtrend_col.append('High')
    else:
        crowdtrend_col.append('Very High')

# ------------------ Comfort ------------------
comfort_col = []
for i in range(n_rows):
    f = footfall_col[i]
    level = 0
    if f < 20:
        level = 0
    elif f < 40:
        level = 1
    elif f < 60:
        level = 2
    else:
        level = 3
    
    if weather_col[i] in ['Rainy','Overcast'] and hour_col[i]>=18:
        level += 1
    if humidity_col[i] > 75:
        level += 1
    if temperature_col[i] > 35:
        level -= 1
    if traffic_col[i]=='High':
        level += 1
    if parking_col[i]=='Full':
        level += 1
    level = max(0, min(level,3))
    
    if (f <= user_max_crowd_col[i] and 
        level <= noise_map[user_noise_tolerance_col[i]] and
        temperature_col[i]>=user_preferred_temp_col[i]-2 and temperature_col[i]<=user_preferred_temp_col[i]+2 and
        humidity_col[i]>=user_preferred_humidity_col[i]-5 and humidity_col[i]<=user_preferred_humidity_col[i]+5):
        comfort_col.append('Comfortable')
    else:
        comfort_col.append(['Calm','Moderate','Noisy','Overloaded'][level])

# ------------------ Build DataFrame ------------------
df = pd.DataFrame({
    'Day': day_col,
    'Hour': hour_col,
    'Weekend': weekend_col,
    'Weather': weather_col,
    'Temperature': temperature_col,
    'Humidity': humidity_col,
    'Cloudiness': cloudiness_col,
    'Event': event_col,
    'Holiday': holiday_col,
    'Festival': festival_col,
    'Season': season_col,
    'Offer': offer_col,
    'Footfall': footfall_col,
    'CrowdDensity': crowdtrend_col,
    'Ratings': ratings_col,
    'Cuisine': cuisine_col,
    'Traffic': traffic_col,
    'Parking': parking_col,
    'ParkingDistance_m': parking_distance_col,
    'NearbyAttractions': attractions_col,
    'AirQuality_AQI': aqi_col,
    'User_Max_Crowd': user_max_crowd_col,
    'User_Noise_Tolerance': user_noise_tolerance_col,
    'User_Preferred_Temp': user_preferred_temp_col,
    'User_Preferred_Humidity': user_preferred_humidity_col,
    'User_Accessibility': user_accessibility_col,
    'User_Seating': user_seating_col,
    'Comfort': comfort_col
})

# ------------------ Save CSV ------------------
df.to_csv('context_crowd_comfort.csv', index=False)
print("Enhanced dataset generated: 'context_crowd_comfort.csv'")

