
import os
import json
import sys
import logging
from dotenv import load_dotenv
import google.generativeai as palm

# Load environment variables from the .env file
def load_configuration():
    """Load the API key from the .env file."""
    load_dotenv()

    palm_api_key = os.getenv("PALM_API_KEY")
    if not palm_api_key:
        logging.error("PALM_API_KEY is missing from .env file.")
        sys.exit(1)

    return palm_api_key

# Configure the API client with the API key
def configure_palm(api_key):
    """Configure the Google Palm API client."""
    palm.configure(api_key=api_key)

# Initialize the model
def initialize_model():
    """Initialize the generative model."""
    try:
        model = palm.GenerativeModel(model_name="gemini-1.5-flash-8b-exp-0924")
        return model
    except Exception as e:
        logging.error(f"Error initializing the model: {str(e)}")
        sys.exit(1)

# Generate a personalized trip itinerary
def generate_itinerary(source, destination, start_date, end_date, budget, model):
    """Generate a personalized trip itinerary based on user input."""
    prompt = f"Generate a structured, personalized trip itinerary for a trip from {source} to {destination} from {start_date} to {end_date} with a budget of {budget}. The itinerary should be broken down by day, with suggested activities, including transportation, food, and budget breakdown."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error generating itinerary: {str(e)}")
        return None

# Parse the generated itinerary into a structured format
def parse_itinerary(itinerary_text):
    """Parse the generated itinerary into a structured format with days and activities."""
    days = []
    day_texts = itinerary_text.split("Day")
    
    # Skip the first empty element before "Day 1"
    for day_text in day_texts[1:]:
        day_info = {}
        lines = day_text.strip().split("\n")
        day_number = lines[0].split(" ")[0]
        day_info["day"] = f"Day {day_number}"

        activities = []
        for line in lines[1:]:
            if line.strip():
                activities.append({"activity": line.strip()})

        day_info["activities"] = activities
        days.append(day_info)

    return {"itinerary": days}

# Main function to run the script
def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Validate the input arguments
    if len(sys.argv) != 6:
        logging.error("Usage: python app.py <source> <destination> <start_date> <end_date> <budget>")
        sys.exit(1)

    # Collect input from command-line arguments
    source, destination, start_date, end_date, budget = sys.argv[1:6]

    # Load configuration and initialize components
    palm_api_key = load_configuration()
    configure_palm(palm_api_key)
    model = initialize_model()

    # Generate itinerary
    itinerary_text = generate_itinerary(source, destination, start_date, end_date, budget, model)

    if itinerary_text:
        # Parse and return the itinerary as JSON
        itinerary = parse_itinerary(itinerary_text)
        print(json.dumps(itinerary, indent=4))
    else:
        logging.error("Failed to generate itinerary.")

if __name__ == "__main__":
    main()
