import pandas as pd
import random
from faker import Faker
from datetime import datetime

# Initialize Faker with US locale
fake = Faker('en_US')
current_year = datetime.now().year

# Expanded list of common U.S. last names
last_names = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz",
    "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris",
    "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan",
    "Cooper", "Peterson", "Bailey", "Reed", "Kelly", "Howard", "Ramos"
]

# Function to generate a birth date and calculate age
def generate_birthdate_and_age():
    birth_year = random.randint(1920, 2005)
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)  # Simplified to 28 to avoid month-end complexities
    birthdate = datetime(birth_year, birth_month, birth_day)
    age = current_year - birth_year
    return birthdate.date(), age

# Generate a set of unique base addresses
base_addresses = [fake.address().replace("\n", ", ") for _ in range(200)]  # 200 unique base addresses

# Function to create variations of base addresses
def create_nearby_address(base_address):
    parts = base_address.split(", ")
    # Slightly change the street number or add a small unit variation
    if len(parts) > 1:
        street = parts[0]
        if street[0].isdigit():  # If the address starts with a number (street number)
            # Change the street number slightly or add a suffix
            street_number = int(street.split()[0])
            variation = random.choice([1, -1, 2, -2])  # Small changes
            street_number += variation
            street = f"{street_number} " + " ".join(street.split()[1:])
        # Add the modified street back
        parts[0] = street
    return ", ".join(parts)

# Generate synthetic data with address variations
data = []
for i in range(10000):
    first_name = fake.first_name()
    last_name = random.choice(last_names)
    full_name = f"{first_name} {last_name}"
    
    # Use a base address or create a variation
    if random.random() > 0.8:  # 20% chance to use a nearby (similar) address
        base_address = random.choice(base_addresses)
        address = create_nearby_address(base_address)
    else:
        address = random.choice(base_addresses)
    
    # Extract the ZIP code from the address and correct it
    zip_code = address.split()[-1] if address.split()[-1].isdigit() else fake.zipcode()
    
    birthdate, age = generate_birthdate_and_age()

    data.append({
        "Name": full_name,
        "Address": address,
        "Pin Code": zip_code,
        "Birth Date": birthdate,
        "Age": age
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_path = "benificiary_data.csv"
df.to_csv(csv_path, index=False)

print(f"CSV file saved to: {csv_path}")
