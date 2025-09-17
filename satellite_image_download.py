import requests
from PIL import Image
from io import BytesIO

# Define your API key
api_key = "AIzaSyCWlfuO4KXwEnVMd8w6Vzl4YvDPpy64S84"

# Define location and parameters
4
latitude = 43.060138101680806
longitude =-87.88807777895609


zoom =  20
size = "640x640"  # Define the size of the image (max 640x640 for free tier)
maptype = "satellite"  # Satellite image type

# Construct the URL
url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={size}&maptype={maptype}&key={api_key}"

# Send a GET request to the API
response = requests.get(url)

# Check if the response is successful
if response.status_code == 200:
    # Open the image using PIL
    image = Image.open(BytesIO(response.content))
    # Display the image
    image.show()
    # Save the image if needed
    image.save("satellite_image.png")
    print("Image downloaded and saved as 'satellite_image.png'")
else:
    print("Failed to retrieve the image:", response.status_code, response.text)

    image.save("satellite_image1.png")

