import datetime
import os
import aiohttp
import discord

from pathlib import Path

from discord.ext import commands, tasks

from matchparse.match_parser import MatchParser
from utils.sheets_manager import GoogleSheetsManager


SILENT_MODE = False

THINKING_EMOJI = '💾'
REPROCESS_EMOJI = '⏪'
NOPROCESS_EMOJI = '❌'  # TODO: use this emoji
DONE_EMOJI = '👍'


DOWNLOADS_DIR = 'downloads'


def _env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name, default):
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_list_int(name, default):
    raw = os.getenv(name)
    if not raw:
        return default
    parts = [p.strip() for p in raw.replace(";", ",").replace(" ", ",").split(",")]
    ids = []
    for p in parts:
        if not p:
            continue
        try:
            ids.append(int(p))
        except ValueError:
            pass
    return ids or default


def _read_secret_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return None


def _get_discord_token():
    # Read only from the docker-compose secret
    token = _read_secret_file('/run/secrets/discord_token')
    if token:
        return token
    return None


# Parameterized settings with sensible defaults
ALLOWED_CHANNEL_IDS = _env_list_int('ALLOWED_CHANNEL_IDS', [1351265799561809920])
SAVE_ICONS = _env_bool('SAVE_ICONS', False)
INITIAL_RECENT_HOURS = _env_int('INITIAL_RECENT_HOURS', 240)
RECURRING_RECENT_HOURS = _env_int('RECURRING_RECENT_HOURS', 1)
print('ENVS', ALLOWED_CHANNEL_IDS, SAVE_ICONS, INITIAL_RECENT_HOURS, RECURRING_RECENT_HOURS) # TODO log this instead


# Define image signatures for validation
IMAGE_SIGNATURES = {
    b'\xFF\xD8\xFF': 'jpg',  # JPEG
    b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A': 'png',  # PNG
}

# Initialize the bot with intents
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
bot = commands.Bot(command_prefix="!", intents=intents)


async def has_bot_reacted(message, emoji):
    for reaction in message.reactions:
        if str(reaction.emoji) == emoji:
            # Check if the bot has added this reaction
            async for user in reaction.users():
                if user.id == bot.user.id:
                    return True
    return False


async def has_anyone_else_reacted(message, emoji):
    for reaction in message.reactions:
        if str(reaction.emoji) == emoji:
            # Check if the bot has added this reaction
            async for user in reaction.users():
                if user.id != bot.user.id:
                    return True
    return False


async def download_image(attachment, fn_prefix):
    """Download an image from a URL and save it to a file."""
    fn = f'{fn_prefix}_{attachment.filename}'
    filepath = Path(DOWNLOADS_DIR) / fn
    if filepath.exists():
        return filepath

    async with aiohttp.ClientSession() as session:
        async with session.get(attachment.url) as response:
            if response.status == 200:
                with open(filepath, 'wb') as f:
                    f.write(await response.read())

    return filepath


async def is_image(attachment):
    """Check if a file is an image by downloading the first few bytes and checking its signature."""
    if not attachment.content_type or not attachment.content_type.startswith('image/'):
        return False

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as response:
                if response.status == 200:
                    header = await response.content.read(32)  # Read the first 32 bytes
                    for signature, _ in IMAGE_SIGNATURES.items():
                        if header.startswith(signature):
                            return True
    except Exception:
        pass
    return False


async def send_message(channel, content=None, embed=None, file=None):
    """
    Send a message to a specified channel.

    Args:
        channel (discord.TextChannel): The channel to send the message to.
        content (str, optional): The text content of the message.
        embed (discord.Embed, optional): An embed to include in the message.
        file (discord.File, optional): A file to attach to the message.

    Returns:
        discord.Message: The message that was sent.
    """
    if SILENT_MODE:
        return None

    try:
        message = await channel.send(content=content, embed=embed, file=file)
        return message
    except Exception as e:
        print(f"Failed to send message: {e}")
        return None


async def process_recent_messages(hours=12):
    """
    Fetch messages from the past 24 hours in ALLOWED_CHANNEL_IDS and process any images.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    time_threshold = now - datetime.timedelta(hours=hours)

    for channel_id in ALLOWED_CHANNEL_IDS:
        channel = bot.get_channel(channel_id)
        if not channel or not isinstance(channel, discord.TextChannel):
            print(f"Channel {channel_id} not found or is not a text channel.")
            continue

        print(f"Fetching messages from channel: {channel.name}")
        async for message in channel.history(after=time_threshold):
            is_bot_done = await has_bot_reacted(message, DONE_EMOJI)
            is_reprocess = await has_anyone_else_reacted(message, REPROCESS_EMOJI)
            is_noprocess = await has_anyone_else_reacted(message, DONE_EMOJI)  # TODO: use noprocess emoji
            if is_noprocess:
                continue
            if is_bot_done and not is_reprocess:
                continue
            for i, attachment in enumerate(message.attachments):
                try:
                    await process_image_attachment(attachment, message, i)
                except Exception:
                    print(f'Failed on message: {message}')


@bot.event
async def on_ready():
    """Event handler for when the bot is ready."""
    print(f'Logged in as {bot.user}')

    await process_recent_messages(hours=INITIAL_RECENT_HOURS)


@tasks.loop(hours=RECURRING_RECENT_HOURS)
async def recurring_reprocess():
    await process_recent_messages(hours=RECURRING_RECENT_HOURS)


@recurring_reprocess.before_loop
async def before_recurring_task():
    await bot.wait_until_ready()  # Wait until the bot is fully ready


@bot.event
async def on_message(message):
    """Event handler for when a message is sent."""
    if message.channel.id in ALLOWED_CHANNEL_IDS and message.attachments:
        for i, attachment in enumerate(message.attachments):
            await process_image_attachment(attachment, message, i)

    # Ensure commands are processed
    await bot.process_commands(message)


async def process_image_attachment(attachment, parent_message, nth_attachment):
    is_img = await is_image(attachment)
    if not is_img:
        await send_message(parent_message.channel,
                           content=f'{parent_message.author.mention}, the file you uploaded is not a valid image')
        return None

    await add_reaction(parent_message, THINKING_EMOJI)

    fn_prefix = f'{parent_message.id}'
    if nth_attachment > 0:
        fn_prefix = f'{fn_prefix}_{nth_attachment}'

    filepath = await download_image(attachment, fn_prefix)
    parser = await process_image(filepath)
    df = parser.to_df()
    await upload_df(df, parent_message, filepath.name)

    await add_reaction(parent_message, DONE_EMOJI)
    await remove_reaction(parent_message, THINKING_EMOJI)
    # await add_reaction(parent_message, REPROCESS_EMOJI)


async def add_reaction(message, emoji):
    if SILENT_MODE:
        return
    print(f'Adding reaction {emoji}')
    await message.add_reaction(emoji)


async def remove_reaction(message, emoji):
    if SILENT_MODE:
        return
    print(f'Removing emoji: {emoji}')
    await message.remove_reaction(emoji, bot.user)


def get_message_metadata(message):
    timestamp = message.created_at.isoformat()
    author = message.author.name
    channel = message.channel.name
    server = message.guild.name if message.guild else "Direct Message"

    return timestamp, author, channel, server


async def process_image(filepath, upload=True):
    parser = MatchParser(filepath)
    parser.run(output_dir='output/', is_save_icons=SAVE_ICONS)

    return parser

async def upload_df(df, message, filename):
    timestamp, author, channel, server = get_message_metadata(message)

    original_columns = list(df.columns)
    df['timestamp'] = timestamp
    df['message_id'] = message.id
    df['author'] = author
    df['channel'] = channel
    df['server'] = server
    df['filename'] = filename
    metadata_columns = ['timestamp', 'message_id',
                        'author', 'channel', 'server',
                        'filename']

    spreadsheet_manager = GoogleSheetsManager()  # TODO: might want to move this elsewhere so we dont load auth every time
    spreadsheet_manager.upload_df(df[metadata_columns + original_columns])


def main():
    """Start the bot."""
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    token = _get_discord_token()
    if not token:
        raise RuntimeError('Discord token not provided. Mount docker secret "discord_token" to /run/secrets/discord_token')
    bot.run(token)


if __name__ == "__main__":
    main()