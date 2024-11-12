import requests
import re

# Set up IMDB API key
API_KEY = "YOUR_API_KEY"

# Define URL for Blue Bloods episode list
url = f"https://api.imdb.com/v3/search/title?keywords=Blue+Bloods&title_type=tv_series&count=100&apiKey={API_KEY}"

# Send GET request to retrieve episode list
response = requests.get(url)

# Parse JSON response
data = response.json()

filtered_episodes = []
for episode in data["results"]:
    # Extract video codec (AVC High@L3.1) and audio codec (AAC LC)
    video_codec = episode["videoCodec"]
    audio_codec = episode["audioCodec"]

    if video_codec == "AVC High@L3.1" and audio_codec == "AAC LC":
        filtered_episodes.append(episode)

relevant_data = []
for episode in filtered_episodes:
    # Use regular expressions to extract metadata (e.g., title, air date, summary)
    metadata = {}
    metadata["title"] = re.search(r"<title>(.*?)</title>", episode["html"]).group(1)
    metadata["air_date"] = re.search(r"Air Date: (.*?)<", episode["html"]).group(1)
    metadata["summary"] = re.search(r"Summary: (.*?)<", episode["html"]).group(1)

    relevant_data.append(metadata)

for metadata in relevant_data:
    print(metadata)
