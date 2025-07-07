# Use a lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy your files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run the bot
CMD ["python", "bot.py"]