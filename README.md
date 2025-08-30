# AlphaBot

AlphaBot is a Discord bot that watches configured channels for match images, parses them, and exports data to a Google Sheet. It runs in Docker and persists downloaded images and parsed outputs to Docker volumes.

NOTE: This project was setup over a weekend. Apoologies for the mess

## Getting Started

Prerequisites
- Make sure Docker is installed. If you don’t have it, install from: https://docs.docker.com/get-docker/

Setup
1) Copy the example env and adjust as needed
   - `cp .env.example .env`
   - Edit `.env` to set channel IDs and behavior (see Config below).

2) Provide required secrets
   - Discord bot token → create file: `./secrets/discord_token` containing only your bot token.
   - Google Service Account credentials JSON → place at: `./secrets/google_api_creds.json`.
     - Ensure the Service Account has access to the target Google Sheet.

3) Build and run
   - `docker compose up --build -d`

The bot will begin scanning the configured channels for recent images and then continue on a recurring interval.

## Configuration

To modify the bot behavior, create an `.env` file to inject Environment variables into your containers:
- `ALLOWED_CHANNEL_IDS`: Comma or semicolon separated list of Discord channel IDs. Default: `1351265799561809920`
  * To find the Channel ID, right click the discord channel and click `Copy Channel ID`
- `SAVE_ICONS`: Whether to save cropped icons detected during parsing. Accepts `true/false`. Default: `false`
- `INITIAL_RECENT_HOURS`: Hours of history to scan on startup. Default: `48`
- `RECURRING_RECENT_HOURS`: Hours of history to scan on each recurring pass. Default: `1`
- `GOOGLE_SPREADSHEET_NAME`: Target Google Sheet name. The Google Sheet should have a sheet called `RAW` Default: `AlphaBot Match Data v1`

Required secrets (mounted as Docker secrets):
- `./secrets/discord_token` → available in the container at `/run/secrets/discord_token`
- `./secrets/google_api_creds.json` → available in the container at `/run/secrets/google_api_creds`

## Viewing Images

There is a simple web app hosted on [localhost:8080](http://localhost:8080/) to browse the download/output images

## Volumes and Data

This compose file defines two named volumes that persist data outside the container:
- `download` → mounted at `/app/downloads` (original image downloads)
- `output` → mounted at `/app/output` (parsed artifacts/results)

To find the host filesystem path for a volume, inspect it and check the `Mountpoint` field. By default, Compose creates names as `<project>_download` and `<project>_output` where `<project>` is typically the folder name `alphabot`.

Examples
- Get the host path for downloads: `docker volume inspect alphabot_download --format '{{ .Mountpoint }}'`
- Get the host path for output:    `docker volume inspect alphabot_output --format '{{ .Mountpoint }}'`

Once you have the host path, you can browse files directly with your file explorer or via the terminal.

## Notes

- The bot requires the Discord Message Content intent enabled in your bot settings for reading attachments and content.
- Ensure the Google Service Account email is shared on the target spreadsheet specified by `GOOGLE_SPREADSHEET_NAME`.
