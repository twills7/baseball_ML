import datetime
import os
import discord
from discord import app_commands
from dotenv import load_dotenv
from google import genai
from mlb_data import csv_to_dataframe, download_and_rename

download_and_rename("batter")
download_and_rename("pitcher")

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
# genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model (using 1.5-pro free tier model)
model = genai.Client()

# Discord bot setup
intents = discord.Intents.default()
discord_client = discord.Client(intents=intents)
tree = app_commands.CommandTree(discord_client)

# Fetch mlb data
today = datetime.date.today().isoformat()
mlb_batters_df = csv_to_dataframe(f"data_snapshots/batter_{today}.csv")
mlb_pitchers_df = csv_to_dataframe(f"data_snapshots/pitcher_{today}.csv")

mlb_batters_df.to_string()
mlb_pitchers_df.to_string()

@discord_client.event
async def on_ready():
    print(f"Logged in as {discord_client.user}")
    await tree.sync()
    print("Slash commands synced.")

@tree.command(name="ask", description="Ask a sports question and get an AI-powered answer.")
async def ask_command(interaction: discord.Interaction, question: str):
    await interaction.response.defer()

    try:
        # Call Gemini API
        response = model.models.generate_content(
            model="gemini-2.5-flash", 
            contents= f"You are a smart sports analytics AI discord bot helping people with all betting decisions."
                      f" You have access to the latest MLB data and can answer questions about players, teams, and stats: \n\n"
                        f"MLB Batters Data:\n{mlb_batters_df.to_string(index=False)}\n\n"
                        f"MLB Pitchers Data:\n{mlb_pitchers_df.to_string(index=False)}\n\n"
                      f"Answer this question in one sentence: {question}"

        )
        answer = response.text  # Gemini outputs plaintext directly

        if len(answer) > 2000:
            answer = answer[:1997] + "..."

        await interaction.followup.send(answer)

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        await interaction.followup.send("⚠️ There was an error getting the answer from Gemini.")

# Run the bot
discord_client.run(DISCORD_TOKEN)

if __name__ == "__main__":
    discord_token = os.getenv("DISCORD_TOKEN")
    discord_client.run(discord_token)