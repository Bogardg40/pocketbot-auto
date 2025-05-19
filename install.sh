#!/bin/bash

echo "⏳ Updating server..."
apt update && apt upgrade -y

echo "⏳ Installing dependencies..."
apt install -y python3 python3-pip unzip wget git curl

echo "⏳ Installing Google Chrome..."
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt install -y ./google-chrome-stable_current_amd64.deb || apt --fix-broken install -y

echo "⏳ Installing ChromeDriver..."
CHROME_VERSION=$(google-chrome --version | grep -oP '[0-9]+' | head -1)
DRIVER_VERSION=$(curl -s https://chromedriver.storage.googleapis.com/LATEST_RELEASE_${CHROME_VERSION})
wget https://chromedriver.storage.googleapis.com/${DRIVER_VERSION}/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
mv chromedriver /usr/bin/chromedriver
chmod +x /usr/bin/chromedriver
rm chromedriver_linux64.zip

echo "⏳ Installing Python packages..."
pip3 install selenium undetected-chromedriver

echo "⏳ Downloading your bot script..."
rm -rf /root/pocketbot
git clone https://github.com/Bogardg40/pocketbot-auto.git /root/pocketbot
cd /root/pocketbot

echo "✅ Starting the bot..."
python3 bot.py
