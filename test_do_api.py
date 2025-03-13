import digitalocean
import os

# Load the token
token = os.getenv("DIGITALOCEAN_TOKEN", "dop_v1_ea6e3f6a198c899eb168d56133c610b431f9870927571d2c7c10a018ba82cb99")
manager = digitalocean.Manager(token=token)

# List Droplets
droplets = manager.get_all_droplets()
for droplet in droplets:
    print(f"Droplet: {droplet.name}, IP: {droplet.ip_address}, Status: {droplet.status}")