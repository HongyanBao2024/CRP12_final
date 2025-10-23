import pandas as pd
from datetime import datetime
from meteostat import Monthly, Stations
import holidays
from itertools import product

def get_avg_temperature(country, year, month):
    try:
        if country not in country_coords:
            return None

        lat, lon = country_coords[country]
        stations = Stations().nearby(lat, lon).fetch(5)

        for station_id in stations.index:
            data = Monthly(station_id, datetime(year, month, 1), datetime(year, month, 1)).fetch()
            if not data.empty:
                if 'tavg' in data.columns and pd.notna(data['tavg'].iloc[0]):
                    return round(data['tavg'].iloc[0], 1)
                elif 'tmin' in data.columns and 'tmax' in data.columns:
                    return round((data['tmin'].iloc[0] + data['tmax'].iloc[0]) / 2, 1)
        return None
    except Exception as e:
        print(f"Error for {country} {year}-{month}: {e}")
        return None

def get_holiday_countryin(country_codes,years,months):
    # Preload holidays for all countries and years
    all_holidays = {}
    for country, code in country_codes.items():
        all_holidays[country] = holidays.country_holidays(code, years=years)

    # Collect all holidays into records list
    records = []
    for country in country_codes.keys():
        for year in years:
            for month in months:
                hols = [
                    name for date, name in all_holidays[country].items()
                    if date.year == year and date.month == month
                ]
                holiday_str = ", ".join(sorted(set(hols))) if hols else "None"
                records.append({
                    "Country": country,
                    "Year": year,
                    "Month": month,
                    "Holiday": holiday_str
                })

    df_grouped = pd.DataFrame(records)

    # Save to Excel
    df_grouped.to_excel("inputoutput/grouped_holidays_2020_2024_with_none.xlsx", index=False)
    return df_grouped

def get_holiday_countryout(years,months):
    # Define Vietnam holiday mapping manually
    vietnam_holidays = {
        2020: {
            1: ["New Year's Day", "Vietnamese New Year (Tet)"],
            4: ["Hung Kings Commemoration Day"],
            5: ["Reunification Day", "Labour Day"],
            9: ["National Day"]
        },
        2021: {
            1: ["New Year's Day"],
            2: ["Vietnamese New Year (Tet)"],
            4: ["Hung Kings Commemoration Day"],
            5: ["Reunification Day", "Labour Day"],
            9: ["National Day"]
        },
        2022: {
            1: ["New Year's Day"],
            2: ["Vietnamese New Year (Tet)"],
            4: ["Hung Kings Commemoration Day"],
            5: ["Reunification Day", "Labour Day"],
            9: ["National Day"]
        },
        2023: {
            1: ["New Year's Day", "Vietnamese New Year (Tet)"],
            4: ["Hung Kings Commemoration Day"],
            5: ["Reunification Day", "Labour Day"],
            9: ["National Day"]
        },
        2024: {
            1: ["New Year's Day"],
            2: ["Vietnamese New Year (Tet)"],
            4: ["Hung Kings Commemoration Day"],
            5: ["Reunification Day", "Labour Day"],
            9: ["National Day"]
        },
    }

    # Generate DataFrame with all months from 2020 to 2024
    records = []
    for year in years:
        for month in months:
            holidays = vietnam_holidays.get(year, {}).get(month, [])
            records.append({
                "Year": year,
                "Month": month,
                "Country": "Vietnam",
                "Holiday": ", ".join(holidays) if holidays else "None"
            })

    vietnam_df = pd.DataFrame(records)

    # Save to Excel
    vietnam_df.to_excel("inputoutput/vietnam_holidays_2020_2024.xlsx", index=False)

    return vietnam_df

# Load your dataset (replace with your file)
df = pd.read_excel('inputoutput/input.xlsx')  # e.g. df = pd.read_excel("input.xlsx")


# Ensure date parsing
df['Month_Year'] = pd.to_datetime(df['Month_Year'])

# Extract year & month
df['Year'] = df['Month_Year'].dt.year
df['Month'] = df['Month_Year'].dt.month

# Initialize empty columns
df['Temperature'] = None

# Country → representative city coordinates
country_coords = {
    'New Zealand': (-36.8485, 174.7633),     # Auckland
    'Australia': (-33.8688, 151.2093),       # Sydney
    'United States': (40.7128, -74.0060),    # New York
    'United Kingdom': (51.5074, -0.1278),    # London
    'Canada': (43.6510, -79.3470),           # Toronto
    'Indonesia': (-6.2088, 106.8456),        # Jakarta
    'Malaysia': (3.1390, 101.6869),          # Kuala Lumpur
    'Vietnam': (21.0285, 105.8544),          # Hanoi
}

# Apply temperature and holiday data
for i, row in df.iterrows():
    country = row['Country']
    year = row['Year']
    month = row['Month']
    date = row['Month_Year']

    # 🌡️ Temperature
    df.at[i, 'Temperature'] = get_avg_temperature(country, year, month)

# ✅ Save output
# df.drop(columns=['Year', 'Month'], inplace=True)
# df.to_excel("inputoutput/enriched_output.xlsx", index=False)
print("Data enriched and saved to 'enriched_output.xlsx'")

# Define countries and their codes
country_codes = {
    "New Zealand": "NZ",
    "Australia": "AU",
    "Indonesia": "ID",
    "Malaysia": "MY",
    
}
years = range(2020, 2025)
months = range(1, 13)

df_grouped = get_holiday_countryin(country_codes,years,months)
vietnam_df = get_holiday_countryout(years,months)
# df = pd.read_excel('inputoutput/input.xlsx')  # e.g. df = pd.read_excel("input.xlsx")

# ------------------------------------
# 2. Load or create the df_grouped with holidays
# ------------------------------------
# Make sure you already generated this using the `holidays` package and filled "None" for months with no holidays

# Sample structure of df_grouped

# Ensure column names match
# Required columns: Country, Year, Month, Holiday

df_grouped_extended = pd.concat([df_grouped, vietnam_df], ignore_index=True)
# ------------------------------------
# 3. Merge on Country + Year + Month
# ------------------------------------
df_merged = pd.merge(df, df_grouped_extended, on=["Country", "Year", "Month"], how="left")
# Assuming your DataFrame is called df_grouped_extended or similar
df_merged['promo'] = df_merged['Holiday'].apply(lambda x: 0 if x == 'None' else 1)

# ------------------------------------
# 4. Save or preview the result
# Save to Excel
df_merged.drop(columns=['Year', 'Month'], inplace=True)
df_merged.to_excel("inputoutput/enriched_output.xlsx", index=False)
