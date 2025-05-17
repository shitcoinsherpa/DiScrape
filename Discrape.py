import gradio as gr
import requests
import json
import os
import pandas as pd
from datetime import datetime
import cryptography.fernet as fernet
import base64
import hashlib
from typing import Dict, List, Optional, Tuple, Union
import time
import re
from PIL import Image
import io
import concurrent.futures
import logging
import random
import sys
import tempfile
import shutil
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("discord_scraper")

# Constants
CONFIG_DIR = os.path.expanduser("~/.discrape")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
KEY_FILE = os.path.join(CONFIG_DIR, "key.txt")

# ASCII Art for hacker vibe (ASCII-only version)
DISCRAPE_ASCII = r"""
 ____  _  ____   ___  ____    __    ____  ____ 
(  _ \(_)/ ___) / __)(  _ \  /__\  (  _ \( ___)
 )(_) ) (\__  \( (__  )   / /(__)\  )___/ )__) 
(____/(_)(____/ \___)(__\_)(__)(__)(__) (____) 
> Infiltrate. Extract. Analyze.        v1.0.0 <
"""

# Matrix-like symbols for animations
MATRIX_CHARS = "!@#$%^&*()_+-=[]{}|;:,./<>?~`abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

class TokenEncryption:
    """Handle encryption and decryption of Discord tokens"""
    
    def __init__(self):
        self._ensure_dir_exists()
        self._load_or_create_key()
        
    def _ensure_dir_exists(self):
        """Ensure the config directory exists"""
        if not os.path.exists(CONFIG_DIR):
            os.makedirs(CONFIG_DIR)
            print(f"[SYSTEM] Created configuration directory: {CONFIG_DIR}")
            
    def _load_or_create_key(self):
        """Load existing key or create a new one"""
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, "rb") as f:
                key = f.read()
            print(f"[SYSTEM] Encryption key loaded")
        else:
            # Generate a key from a password
            password = "discrape_secure_" + datetime.now().strftime("%Y%m%d")
            key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
            with open(KEY_FILE, "wb") as f:
                f.write(key)
            print(f"[SYSTEM] New encryption key generated")
        
        self.cipher = fernet.Fernet(key)
    
    def encrypt_token(self, token: str) -> str:
        """Encrypt a token"""
        return self.cipher.encrypt(token.encode()).decode()
    
    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt a token"""
        try:
            return self.cipher.decrypt(encrypted_token.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt token: {e}")
            return ""

class ConfigManager:
    """Manage configuration persistence"""
    
    def __init__(self, encryption: TokenEncryption):
        self.encryption = encryption
        self._ensure_dir_exists()
        self.config = self._load_config()
        
    def _ensure_dir_exists(self):
        """Ensure the config directory exists"""
        if not os.path.exists(CONFIG_DIR):
            os.makedirs(CONFIG_DIR)
            
    def _load_config(self) -> Dict:
        """Load configuration from file or create default"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                return self._get_default_config()
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            "encrypted_token": "",
            "recent_servers": [],
            "recent_channels": [],
            "rate_limit": 1.0,  # seconds between requests
            "download_folder": os.path.expanduser("~/Downloads/discrape"),
            "theme": "dark"
        }
    
    def save_config(self):
        """Save current configuration to file"""
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f)
            
    def get_token(self) -> str:
        """Get decrypted token"""
        if not self.config["encrypted_token"]:
            return ""
        return self.encryption.decrypt_token(self.config["encrypted_token"])
    
    def set_token(self, token: str):
        """Set and encrypt token"""
        if token:
            self.config["encrypted_token"] = self.encryption.encrypt_token(token)
            self.save_config()
            
    def add_recent_server(self, server_id: str, server_name: str = None):
        """Add a server to recent servers list"""
        servers = self.config["recent_servers"]
        # Check if server already exists
        for i, server in enumerate(servers):
            if server["id"] == server_id:
                # Move to top if exists
                servers.pop(i)
                break
                
        # Add to top of list
        servers.insert(0, {"id": server_id, "name": server_name or server_id})
        # Limit to 10 recent servers
        self.config["recent_servers"] = servers[:10]
        self.save_config()
        
    def add_recent_channel(self, channel_id: str, channel_name: str = None):
        """Add a channel to recent channels list"""
        channels = self.config["recent_channels"]
        # Check if channel already exists
        for i, channel in enumerate(channels):
            if channel["id"] == channel_id:
                # Move to top if exists
                channels.pop(i)
                break
                
        # Add to top of list
        channels.insert(0, {"id": channel_id, "name": channel_name or channel_id})
        # Limit to 10 recent channels
        self.config["recent_channels"] = channels[:10]
        self.save_config()
    
    def get_download_folder(self) -> str:
        """Get the download folder, creating it if it doesn't exist"""
        folder = self.config.get("download_folder", os.path.expanduser("~/Downloads/discrape"))
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder
    
    def set_download_folder(self, folder: str):
        """Set the download folder"""
        self.config["download_folder"] = folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.save_config()
        
    def set_rate_limit(self, rate_limit: float):
        """Set the rate limit"""
        self.config["rate_limit"] = float(rate_limit)
        self.save_config()
        
    def get_rate_limit(self) -> float:
        """Get the rate limit"""
        return float(self.config.get("rate_limit", 1.0))

class DiscordAPI:
    """Interface with Discord API"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.base_url = "https://discord.com/api/v10"
        
    def _get_headers(self) -> Dict:
        """Get headers with authorization token"""
        token = self.config.get_token()
        if not token:
            raise ValueError("Discord token not set")
        return {
            "Authorization": token,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def _handle_rate_limit(self, response):
        """Handle rate limiting by sleeping if needed"""
        if response.status_code == 429:
            retry_after = response.json().get("retry_after", 5)
            logger.warning(f"Rate limited. Waiting {retry_after} seconds")
            time.sleep(retry_after)
            return True
        return False
    
    def get_current_user(self) -> Dict:
        """Get current user info"""
        response = requests.get(f"{self.base_url}/users/@me", headers=self._get_headers())
        if self._handle_rate_limit(response):
            return self.get_current_user()
        
        response.raise_for_status()
        return response.json()
    
    def get_server_info(self, server_id: str) -> Dict:
        """Get server information"""
        response = requests.get(f"{self.base_url}/guilds/{server_id}", headers=self._get_headers())
        if self._handle_rate_limit(response):
            return self.get_server_info(server_id)
        
        response.raise_for_status()
        return response.json()
    
    def get_server_channels(self, server_id: str) -> List[Dict]:
        """Get all channels in a server"""
        response = requests.get(f"{self.base_url}/guilds/{server_id}/channels", headers=self._get_headers())
        if self._handle_rate_limit(response):
            return self.get_server_channels(server_id)
        
        response.raise_for_status()
        return response.json()
    
    def get_channel_messages(self, channel_id: str, limit: int = 100, before: Optional[str] = None) -> List[Dict]:
        """Get messages from a channel"""
        params = {"limit": limit}
        if before:
            params["before"] = before
            
        response = requests.get(
            f"{self.base_url}/channels/{channel_id}/messages", 
            headers=self._get_headers(),
            params=params
        )
        
        if self._handle_rate_limit(response):
            return self.get_channel_messages(channel_id, limit, before)
        
        response.raise_for_status()
        # Apply rate limit
        time.sleep(self.config.get_rate_limit())
        return response.json()
    
    def download_attachment(self, url: str, filename: str) -> str:
        """Download attachment from URL"""
        save_path = os.path.join(self.config.get_download_folder(), filename)
        
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            f.write(response.content)
            
        return save_path

class DiscordScraper:
    """Main scraper logic"""
    
    def __init__(self, api: DiscordAPI, config: ConfigManager):
        self.api = api
        self.config = config
        
    def scrape_channel(self, channel_id: str, max_messages: int = 1000, progress_callback=None) -> List[Dict]:
        """Scrape messages from a channel"""
        all_messages = []
        before = None
        batch_size = min(100, max_messages)  # Discord API limit is 100
        
        while len(all_messages) < max_messages:
            try:
                messages = self.api.get_channel_messages(channel_id, batch_size, before)
                if not messages:
                    break
                    
                all_messages.extend(messages)
                
                # Update progress if callback provided
                if progress_callback:
                    progress_callback(len(all_messages), max_messages)
                
                # Set before to the ID of the last message for pagination
                before = messages[-1]["id"]
                
                # If we got fewer messages than requested, we've reached the end
                if len(messages) < batch_size:
                    break
            except Exception as e:
                logger.error(f"Error scraping channel {channel_id}: {e}")
                break
                
        return all_messages[:max_messages]  # Ensure we don't exceed max_messages
    
    def scrape_multiple_channels(self, channel_ids: List[str], max_messages_per_channel: int = 1000, 
                                progress_callback=None) -> Dict[str, List[Dict]]:
        """Scrape messages from multiple channels"""
        results = {}
        total_channels = len(channel_ids)
        
        for i, channel_id in enumerate(channel_ids):
            if progress_callback:
                progress_callback(i, total_channels, channel_id)
                
            results[channel_id] = self.scrape_channel(
                channel_id, 
                max_messages_per_channel,
                lambda current, total: progress_callback(i, total_channels, channel_id, current, total) if progress_callback else None
            )
            
        return results
    
    def download_attachments(self, messages: List[Dict], attachment_types: List[str] = None, 
                           progress_callback=None) -> Dict[str, List[str]]:
        """Download attachments from messages"""
        if attachment_types is None:
            attachment_types = ["png", "jpg", "jpeg", "gif", "mp4", "webm", "mp3", "wav", "pdf"]
            
        attachment_urls = {}
        for msg in messages:
            if "attachments" in msg and msg["attachments"]:
                # Handle both object and string representations of attachments
                attachments = msg["attachments"]
                if isinstance(attachments, str):
                    try:
                        # If it's a JSON string, parse it
                        attachments = json.loads(attachments)
                    except:
                        logger.error(f"Failed to parse attachments JSON: {attachments}")
                        attachments = []
                
                if not isinstance(attachments, list):
                    # If it's not a list after potential JSON parsing, make it a list
                    attachments = [attachments]
                
                for attachment in attachments:
                    if not attachment:
                        continue
                        
                    # Check file type
                    file_ext = None
                    if isinstance(attachment, dict):
                        if "url" in attachment and attachment["url"]:
                            file_ext = attachment["url"].split(".")[-1].lower()
                        elif "filename" in attachment and attachment["filename"]:
                            file_ext = attachment["filename"].split(".")[-1].lower()
                    
                    if not file_ext:
                        logger.warning(f"Could not determine file extension for attachment in msg {msg['id']}: {attachment}")
                        continue
                        
                    if not attachment_types or file_ext in attachment_types:
                        msg_id = msg["id"]
                        if msg_id not in attachment_urls:
                            attachment_urls[msg_id] = []
                        attachment_urls[msg_id].append({
                            "url": attachment.get("url", ""),
                            "filename": attachment.get("filename", ""),
                            "content_type": attachment.get("content_type", ""),
                            "message": msg
                        })
                        logger.info(f"Found matching attachment in message {msg_id}: {attachment.get('filename', '')}")
        
        # Log how many attachments we found
        total_attachments = sum(len(attachments) for attachments in attachment_urls.values())
        logger.info(f"Found {total_attachments} total attachments matching filters: {attachment_types}")
        
        # Download the attachments
        downloaded_files = {}
        total = sum(len(attachments) for attachments in attachment_urls.values())
        current = 0
        
        for msg_id, attachments in attachment_urls.items():
            downloaded_files[msg_id] = []
            for attachment in attachments:
                try:
                    url = attachment["url"]
                    if not url:
                        logger.error(f"Skipping download for attachment in msg {msg_id}: URL is missing.")
                        continue
                        
                    filename = attachment["filename"]
                    if not filename:
                        # Use the determined file_ext to build a filename if original is missing
                        file_ext = url.split(".")[-1].lower()
                        filename = f"attachment_{msg_id}.{file_ext}"
                    
                    # Download the attachment
                    safe_filename = f"{msg_id}_{filename}"
                    file_path = self.api.download_attachment(url, safe_filename)

                    # Track the downloaded file
                    if msg_id not in downloaded_files:
                        downloaded_files[msg_id] = []
                        
                    downloaded_files[msg_id].append({
                        "path": file_path, # This is now the correct path from download_attachment
                        "filename": filename, # original filename part
                        "content_type": attachment.get("content_type", ""),
                        "message": attachment["message"]
                    })
                    
                    logger.info(f"Downloaded attachment via API method: {file_path}")
                except Exception as e:
                    logger.error(f"Error processing/downloading attachment {attachment.get('url', 'unknown')} in msg {msg_id}: {e}", exc_info=True)
                
                current += 1
                if progress_callback:
                    progress_callback(current, total)
                    
        return downloaded_files
    
    def extract_image_prompt_pairs(self, messages: List[Dict], bot_ids: List[str] = None) -> List[Dict]:
        """Extract image-prompt pairs, focusing on bot messages (like Midjourney)"""
        if bot_ids is None:
            # Default Midjourney bot ID
            bot_ids = ["936929561302675456"]
            
        pairs = []
        logger.info(f"Extracting image-prompt pairs for bot_ids: {bot_ids}")

        for msg_idx, msg in enumerate(messages):
            msg_id_for_log = msg.get("id", f"unknownMsg_{msg_idx}")
            # Check if message is from a bot we're interested in and has attachments field
            if msg.get("author", {}).get("id") in bot_ids and "attachments" in msg and msg["attachments"]:
                
                current_msg_attachments = msg["attachments"]
                # Robust handling of attachments structure (could be list or JSON string)
                if isinstance(current_msg_attachments, str):
                    try:
                        current_msg_attachments = json.loads(current_msg_attachments)
                        logger.debug(f"Msg {msg_id_for_log}: Parsed attachments string to list.")
                    except json.JSONDecodeError as e:
                        logger.error(f"Msg {msg_id_for_log}: Failed to parse attachments JSON string '{current_msg_attachments}': {e}")
                        current_msg_attachments = [] # Set to empty list on error
                
                if not isinstance(current_msg_attachments, list):
                    logger.warning(f"Msg {msg_id_for_log}: Attachments field is not a list or valid JSON string, actually type: {type(current_msg_attachments)}. Content: {current_msg_attachments}. Skipping attachments for this message.")
                    current_msg_attachments = [current_msg_attachments] # Attempt to treat as single item list if not already list (defensive)

                images = []
                for att_idx, attachment in enumerate(current_msg_attachments):
                    if not isinstance(attachment, dict):
                        logger.warning(f"Msg {msg_id_for_log}, Att {att_idx}: Attachment item is not a dictionary, actual type: {type(attachment)}. Content: {attachment}. Skipping this item.")
                        continue

                    content_type = attachment.get("content_type", "")
                    logger.debug(f"Msg {msg_id_for_log}, Att {att_idx}: Checking attachment with content_type: '{content_type}', filename: '{attachment.get("filename")}'")
                    if content_type.startswith("image/"):
                        images.append({
                            "url": attachment.get("url"), # Ensure .get() for safety
                            "filename": attachment.get("filename"),
                            "width": attachment.get("width"),
                            "height": attachment.get("height")
                        })
                        logger.debug(f"Msg {msg_id_for_log}, Att {att_idx}: Added image attachment: {attachment.get("filename")}")
                
                if not images:
                    logger.debug(f"Msg {msg_id_for_log}: No image attachments found after filtering for this bot message.")
                    continue # Skip if no image attachments found for this bot message
                    
                # Try to extract prompt from the message content
                content = msg.get("content", "")
                prompt = None
                
                match = re.search(r"\*\*(.*?)\*\*", content)
                if match:
                    prompt = match.group(1)
                elif msg.get("referenced_message") and isinstance(msg.get("referenced_message"), dict):
                    prompt = msg.get("referenced_message", {}).get("content", "")
                
                logger.info(f"Msg {msg_id_for_log}: Found pair. Prompt: '{prompt is not None}', Images: {len(images)}")
                pairs.append({
                    "message_id": msg.get("id"),
                    "timestamp": msg.get("timestamp"),
                    "prompt": prompt,
                    "content": content,
                    "images": images,
                    "author": {
                        "id": msg.get("author", {}).get("id"),
                        "username": msg.get("author", {}).get("username"),
                        "discriminator": msg.get("author", {}).get("discriminator", "0000"),
                        "bot": msg.get("author", {}).get("bot", False)
                    }
                })
            # else:
                # logger.debug(f"Msg {msg_id_for_log}: Skipped. Bot ID match: {msg.get('author', {}).get('id') in bot_ids}, Has attachments: {"attachments" in msg and msg["attachments"]}")
                
        logger.info(f"Extraction complete. Found {len(pairs)} image-prompt pairs.")
        return pairs

class GradioInterface:
    """Gradio UI interface"""
    
    def __init__(self):
        self.encryption = TokenEncryption()
        self.config_manager = ConfigManager(self.encryption)
        self.api = DiscordAPI(self.config_manager)
        self.scraper = DiscordScraper(self.api, self.config_manager)
        
        # State variables
        self.current_server_id = None
        self.current_channel_id = None
        self.available_channels = []
        self.scraped_messages = []
        self.scraped_image_prompts = []
        self.downloaded_attachment_details = {} # To store details of successfully downloaded attachments
        
    def matrix_animation(self):
        """Generate a matrix-like animation string"""
        lines = []
        for i in range(3):
            line = ''.join(random.choice(MATRIX_CHARS) for _ in range(70))
            lines.append(line)
        return '\n'.join(lines)
        
    def build_interface(self):
        """Build the Gradio interface"""
        # Instead of requiring a specific theme system, let's use CSS primarily
        # and add minimal theming that works across Gradio versions
        
        # Custom CSS provides the bulk of our styling
        custom_css = """
            body, .gradio-container {
                background-color: #000000 !important;
                color: #00FF00 !important;
                font-family: 'Fira Code', monospace !important;
            }
            
            /* Matrix animation */
            .matrix {
                color: #00FF00 !important;
                font-family: monospace !important;
                animation: fadeInOut 2s infinite;
            }
            @keyframes fadeInOut {
                0% {opacity: 0.3;}
                50% {opacity: 1;}
                100% {opacity: 0.3;}
            }
            
            /* Hacker box */
            .hacker-box {
                border: 1px solid #00FF00 !important;
                padding: 10px !important;
                border-radius: 0px !important;
                background-color: rgba(0, 10, 0, 0.3) !important;
                margin-bottom: 10px !important;
            }
            
            /* Headings */
            h1, h2, h3, h4 {
                color: #00FF00 !important;
                text-shadow: 0 0 5px #00FF00, 0 0 10px #00FF00 !important;
            }
            
            /* Footer */
            .footer {
                text-align: center !important;
                margin-top: 20px !important;
                color: #00FF00 !important;
                font-style: italic !important;
            }
            
            /* Form elements */
            button, .gr-button {
                text-transform: uppercase !important;
                letter-spacing: 1px !important;
                font-weight: 600 !important;
                color: #00FF00 !important;
                background-color: #003300 !important;
                border: 1px solid #00FF00 !important;
            }
            button:hover, .gr-button:hover {
                background-color: #004400 !important;
            }
            
            input, select, textarea, .gr-input, .gr-dropdown {
                background-color: #000000 !important;
                color: #00FF00 !important;
                border: 1px solid #00FF00 !important;
            }
            
            /* Main blocks and containers */
            .gradio-box, .gr-panel, .gr-form, .container {
                background-color: #000000 !important;
                border-color: #333333 !important;
            }
            
            .gradio-box, .gr-box, .gr-panel {
                background-color: #101010 !important;
            }
            
            /* Labels */
            label, .gr-label {
                color: #00FF00 !important;
            }
            
            /* Progress bar */
            progress {
                color: #00FF00 !important;
            }
            
            /* Tabs */
            .tabs {
                background-color: #000000 !important;
            }
            .tab-nav {
                color: #00FF00 !important;
                background-color: #000000 !important;
            }
            .tab-selected {
                color: #ffffff !important;
                background-color: #007700 !important;
            }
            
            /* Tables and dataframes */
            table, .table {
                background-color: #000000 !important;
                color: #00FF00 !important;
                border: 1px solid #00FF00 !important;
            }
            th, td {
                border: 1px solid #333333 !important;
            }
            
            /* Ensure no white backgrounds sneak in */
            * {
                scrollbar-color: #00FF00 #000000 !important;
            }
            .dark {
                background-color: #000000 !important;
            }
        """
        
        # Use the simplest theme initialization that works across versions
        try:
            with gr.Blocks(title="DiScrape", css=custom_css) as app:
                # Title and description
                gr.Markdown(f"```{DISCRAPE_ASCII}```")
                
                with gr.Row():
                    gr.Markdown('<p class="matrix" id="matrix-text">Initializing... System ready...</p>')
                
                # Hidden element for matrix animation updates
                matrix_output = gr.Textbox(visible=False, every=3)
                
                # Tabs for different functions
                with gr.Tabs():
                    # Authentication tab
                    with gr.Tab("> AUTHENTICATION"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('<div class="hacker-box">Enter your Discord access token to begin infiltration sequence.</div>')
                                token_input = gr.Textbox(
                                    label="> DISCORD_TOKEN",
                                    placeholder="Enter credentials...",
                                    type="password",
                                    value=self.config_manager.get_token()
                                )
                                save_token_btn = gr.Button(">> AUTHENTICATE", variant="primary")
                                token_status = gr.Textbox(label="> STATUS", interactive=False)
                                
                                gr.Markdown("""
                                ### SECURITY PROTOCOLS - TOKEN EXTRACTION
                                
                                1. Launch Discord in browser environment
                                2. Activate DevTools (F12)
                                3. Navigate to Network surveillance module
                                4. Force network activity by accessing Discord channel
                                5. Locate authorization header in HTTP requests
                                6. Extract token value
                                
                                **[ ! ] WARNING: TOKEN IS HIGH-SECURITY CLEARANCE. PROTECT AT ALL COSTS.**
                                """)
                    
                    # Server/Channel Configuration tab
                    with gr.Tab("> TARGET ACQUISITION"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('<div class="hacker-box">Specify target server for data extraction operation.</div>')
                                server_id_input = gr.Textbox(
                                    label="> SERVER_ID",
                                    placeholder="Enter target server identifier..."
                                )
                                
                                server_info_btn = gr.Button(">> SCAN SERVER", variant="primary")
                                
                                server_info_output = gr.Textbox(
                                    label="> SERVER_INFO",
                                    interactive=False
                                )
                                
                                server_channel_info = gr.Textbox(
                                    label="> SERVER_CHANNELS",
                                    placeholder="Available channels will be listed here...",
                                    interactive=False,
                                    lines=10
                                )
                                
                                # Simplified: Direct channel entry without add button
                                selected_channels = gr.TextArea(
                                    label="> TARGET_CHANNELS",
                                    placeholder="Enter channel IDs directly, separated by commas...",
                                    interactive=True,
                                    lines=3,
                                    info="Enter one or more channel IDs separated by commas (e.g., 123456789, 987654321)"
                                )
                                
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("""
                                ### TARGET IDENTIFICATION PROCEDURE
                                
                                1. Enable Developer Mode in Discord settings
                                2. Right-click server -> Copy Server ID
                                3. Right-click channel -> Copy Channel ID
                                """)
                    
                    # Message Scraper tab
                    with gr.Tab("> DATA EXTRACTION"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('<div class="hacker-box">Configure extraction parameters for optimal data acquisition.</div>')
                                max_messages = gr.Slider(
                                    label="> MAX_MESSAGES_PER_CHANNEL",
                                    minimum=10,
                                    maximum=1000000,  # Increased to 1 million messages
                                    step=1000,
                                    value=10000
                                )
                                
                                include_attachments = gr.Checkbox(
                                    label="> DOWNLOAD_ATTACHMENTS",
                                    value=True
                                )
                                
                                attachment_types = gr.CheckboxGroup(
                                    label="> ATTACHMENT_TYPES",
                                    choices=["png", "jpg", "jpeg", "gif", "mp4", "webm", "mp3", "wav", "pdf"],
                                    value=["png", "jpg", "jpeg", "gif"]
                                )
                                
                                output_format = gr.Radio(
                                    label="> OUTPUT_FORMAT",
                                    choices=["JSON", "CSV"],  # Removed Excel option to avoid openpyxl dependency
                                    value="CSV"
                                )
                                
                                scrape_btn = gr.Button(">> INITIATE EXTRACTION", variant="primary")
                                
                        with gr.Row():
                            with gr.Column():
                                progress_bar = gr.Progress()
                                status_output = gr.Textbox(
                                    label="> STATUS",
                                    interactive=False
                                )
                                
                                # Button with correct text
                                # Fix the button text - no >> prefix
                                download_btn = gr.Button("DOWNLOAD", variant="primary", elem_id="download_button")
                                
                                # Add file component for downloadable files
                                download_file = gr.File(
                                    label="> DOWNLOAD FILE"
                                )
                                
                                preview_data = gr.Dataframe(
                                    label="> DATA_PREVIEW",
                                    interactive=False,
                                    wrap=True
                                )
                    
                    # Image & Prompt tab
                    with gr.Tab("> IMAGE RECONNAISSANCE"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('<div class="hacker-box">Extract image-prompt pairs from AI image generation channels.</div>')
                                bot_id_input = gr.Textbox(
                                    label="> BOT_IDS",
                                    placeholder="Enter bot identifiers (comma separated)...",
                                    value="936929561302675456"
                                )
                                
                                channel_id_input = gr.Textbox(
                                    label="> CHANNEL_ID",
                                    placeholder="Enter target channel identifier..."
                                )
                                
                                max_images = gr.Slider(
                                    label="> MAX_MESSAGES",
                                    minimum=10,
                                    maximum=1000000,
                                    step=1000,
                                    value=10000
                                )
                                
                                scrape_images_btn = gr.Button(">> EXTRACT IMAGES", variant="primary")
                        
                        with gr.Row():
                            with gr.Column():
                                image_progress = gr.Progress()
                                image_status = gr.Textbox(
                                    label="> STATUS",
                                    interactive=False
                                )
                        
                        with gr.Row():
                            image_gallery = gr.Gallery(
                                label="> IMAGE_RESULTS",
                                show_label=True,
                                elem_id="gallery",
                                columns=4,
                                height=800
                            )
                            
                        with gr.Row():
                            download_image_pairs_btn = gr.Button(">> DOWNLOAD", variant="primary")
                            # Add file component for image data download
                            image_download_file = gr.File(
                                label="> DOWNLOAD IMAGE DATA"
                            )
                    
                    # Settings tab
                    with gr.Tab("> SYSTEM CONFIG"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('<div class="hacker-box">Configure system parameters for optimal performance.</div>')
                                download_folder = gr.Textbox(
                                    label="> DOWNLOAD_DIRECTORY",
                                    placeholder="Enter file storage location...",
                                    value=self.config_manager.get_download_folder()
                                )
                                
                                rate_limit = gr.Slider(
                                    label="> REQUEST_DELAY (SEC)",
                                    minimum=0.5,
                                    maximum=5.0,
                                    step=0.1,
                                    value=self.config_manager.get_rate_limit()
                                )
                                
                                save_settings_btn = gr.Button(">> SAVE CONFIGURATION", variant="primary")
                                settings_status = gr.Textbox(
                                    label="> STATUS",
                                    interactive=False
                                )
                
                # Footer
                gr.Markdown('<div class="footer">DiScrape v1.0.0 | Use with caution. Stay in the shadows.</div>')
                
                # Event handlers
                
                # Matrix animation update
                def update_matrix(text):
                    return self.matrix_animation()
                    
                matrix_output.change(
                    fn=update_matrix,
                    inputs=[matrix_output],
                    outputs=[matrix_output]
                )
                
                # Authentication tab
                save_token_btn.click(
                    fn=self.save_token,
                    inputs=[token_input],
                    outputs=[token_status]
                )
                
                # Server/Channel tab - removed channel selection click handler since we're using direct entry
                server_info_btn.click(
                    fn=self.get_server_info,
                    inputs=[server_id_input],
                    outputs=[server_info_output, server_channel_info]
                )
                
                # Message Scraper tab
                scrape_btn.click(
                    fn=self.scrape_messages,
                    inputs=[selected_channels, max_messages, include_attachments, attachment_types, output_format],
                    outputs=[status_output, preview_data, download_btn],
                )
                
                # FIX 2: Updated download button to use file download component
                download_btn.click(
                    fn=self.download_results,
                    inputs=[output_format],
                    outputs=[status_output, download_file]
                )
                
                # Image & Prompt tab
                scrape_images_btn.click(
                    fn=self.scrape_image_prompts,
                    inputs=[channel_id_input, bot_id_input, max_images],
                    outputs=[image_status, image_gallery, download_image_pairs_btn]
                )
                
                # FIX 2: Updated image download to use file download component
                download_image_pairs_btn.click(
                    fn=self.download_image_pairs,
                    inputs=[],
                    outputs=[image_status, image_download_file]
                )
                
                # Settings tab
                save_settings_btn.click(
                    fn=self.save_settings,
                    inputs=[download_folder, rate_limit],
                    outputs=[settings_status]
                )
                
        except Exception as e:
            logger.error(f"Error creating interface: {e}")
            # If we have a problem with the fancy interface, fall back to a simpler one
            with gr.Blocks(title="DiScrape") as app:
                gr.Markdown("# DiScrape - ERROR")
                gr.Markdown(f"Failed to create the full interface: {str(e)}")
                gr.Markdown("Try updating Gradio with: pip install --upgrade gradio")
                
        return app
    
    def save_token(self, token):
        """Save Discord token"""
        try:
            self.config_manager.set_token(token)
            # Test the token by getting user info
            user_info = self.api.get_current_user()
            return f"[ACCESS GRANTED] Authentication successful. User: {user_info['username']}#{user_info['discriminator']}"
        except Exception as e:
            logger.error(f"Failed to save token: {e}")
            return f"[ACCESS DENIED] Authentication failed: {str(e)}"
    
    def get_server_info(self, server_id):
        """Get server info and channels"""
        try:
            server_info = self.api.get_server_info(server_id)
            self.current_server_id = server_id
            
            # Add to recent servers
            self.config_manager.add_recent_server(server_id, server_info["name"])
            
            # Get channels
            channels = self.api.get_server_channels(server_id)
            
            # Filter for text channels only
            text_channels = [c for c in channels if c["type"] == 0]  # 0 = text channel
            
            # Format channel list as text for easy reading
            channels_text = "Available channels:\n\n"
            for c in text_channels:
                channels_text += f"ID: {c['id']} - Name: {c['name']}\n"
            
            # Save channels for reference
            self.available_channels = {c["id"]: c["name"] for c in text_channels}
            
            # Return server info and formatted channel list
            return (
                f"[TARGET IDENTIFIED]\n"
                f"NAME: {server_info['name']}\n"
                f"ID: {server_id}\n"
                f"OWNER: {server_info['owner_id']}\n"
                f"REGION: {server_info.get('region', 'UNKNOWN')}\n"
                f"MEMBERS: {server_info.get('approximate_member_count', 'UNKNOWN')}\n"
                f"CHANNELS: {len(text_channels)}\n"
                f"STATUS: READY FOR EXTRACTION",
                channels_text
            )
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
            return f"[ERROR] Target acquisition failed: {str(e)}", ""
    
    def scrape_messages(self, selected_channels_text, max_messages, include_attachments, attachment_types, output_format, progress=gr.Progress()):
        """Scrape messages from selected channels"""
        try:
            logger.info(f"Starting scrape_messages. include_attachments: {include_attachments}, attachment_types: {attachment_types}")
            self._clear_temp_files()
            self.downloaded_attachment_details = {}
            
            if not selected_channels_text or selected_channels_text.strip() == "":
                return "[ERROR] No extraction targets selected", None, False
                
            channel_ids = [c.strip() for c in selected_channels_text.split(",")]
            if not channel_ids:
                return "[ERROR] No extraction targets selected", None, False
            
            for channel_id in channel_ids:
                self.config_manager.add_recent_channel(channel_id)
                
            download_folder = self.config_manager.get_download_folder()
            os.makedirs(download_folder, exist_ok=True)
                
            results = {}
            total_channels = len(channel_ids)
            
            for i, channel_id in enumerate(channel_ids):
                progress(i / total_channels, f"[EXTRACTING] Channel {i+1}/{total_channels} messages...")
                
                messages = self.scraper.scrape_channel(
                    channel_id,
                    max_messages,
                    lambda current, total: progress((i + current/total) / total_channels, 
                                                  f"[EXTRACTING] Channel {i+1}/{total_channels}: {current}/{total} messages")
                )
                
                if include_attachments and messages:
                    logger.info(f"Processing attachments for channel {channel_id}. Found {len(messages)} messages.")
                    
                    # Prepare normalized attachment types for filtering
                    # Handles if attachment_types is None (meaning all types) or a list
                    normalized_filter_types = set()
                    if attachment_types: # If specific types are selected
                        normalized_filter_types = set(t.lower() for t in attachment_types)
                        if "jpeg" in normalized_filter_types:
                            normalized_filter_types.add("jpg")
                        if "jpg" in normalized_filter_types:
                            normalized_filter_types.add("jpeg")
                    logger.info(f"Normalized attachment filter types: {normalized_filter_types or 'All'}")

                    # Calculate total potential attachments for better progress
                    # This count is based on attachments *before* type filtering for progress accuracy
                    channel_potential_attachment_count = 0
                    for msg_check in messages:
                        if "attachments" in msg_check and msg_check["attachments"]:
                            current_msg_attachments = msg_check["attachments"]
                            if isinstance(current_msg_attachments, str):
                                try: current_msg_attachments = json.loads(current_msg_attachments)
                                except: current_msg_attachments = []
                            if not isinstance(current_msg_attachments, list): current_msg_attachments = [current_msg_attachments]
                            channel_potential_attachment_count += len([att for att in current_msg_attachments if att])
                    
                    logger.info(f"Channel {channel_id} has {channel_potential_attachment_count} potential attachments.")

                    if channel_potential_attachment_count > 0:
                        current_attachment_processed_count = 0 # For progress within this channel's attachments
                        
                        for msg_idx, msg in enumerate(messages):
                            msg_id = msg.get("id", f"unknownMsgId_{msg_idx}")
                            if "attachments" in msg and msg["attachments"]:
                                raw_attachments_data = msg["attachments"]
                                logger.debug(f"Message {msg_id} raw attachments data: {raw_attachments_data}")

                                processed_attachments_list = raw_attachments_data
                                if isinstance(processed_attachments_list, str):
                                    try: processed_attachments_list = json.loads(processed_attachments_list)
                                    except Exception as json_e:
                                        logger.error(f"JSON parsing error for attachments in msg {msg_id}: {json_e}")
                                        processed_attachments_list = []
                                if not isinstance(processed_attachments_list, list):
                                    processed_attachments_list = [processed_attachments_list]
                                
                                for att_item_idx, att_item in enumerate(processed_attachments_list):
                                    current_attachment_processed_count +=1 # Increment for each item processed for progress
                                    progress_text_suffix = f"item {current_attachment_processed_count}/{channel_potential_attachment_count}"
                                    progress(
                                        (i + 0.5 + (0.5 * current_attachment_processed_count / channel_potential_attachment_count if channel_potential_attachment_count > 0 else 0)) / total_channels,
                                        f"[PROCESSING ATTS] Ch {i+1}/{total_channels}: {progress_text_suffix}"
                                    )

                                    if not att_item or not isinstance(att_item, dict):
                                        logger.warning(f"Skipping invalid attachment item in msg {msg_id}: {att_item}")
                                        continue

                                    file_ext = None
                                    original_filename = att_item.get("filename")
                                    content_type = att_item.get("content_type", "").lower()
                                    url = att_item.get("url")

                                    logger.debug(f"Att item {att_item_idx} in msg {msg_id}: filename='{original_filename}', content_type='{content_type}', url='{url}'")
                                    
                                    if isinstance(att_item, dict):
                                        if "url" in att_item and att_item["url"]:
                                            # Basic URL extension check (will be refined)
                                            parts = att_item["url"].split(".")
                                            if len(parts) > 1:
                                                file_ext = parts[-1].lower().split('?')[0] # Get part after last dot, remove query params
                                                if not (1 < len(file_ext) < 6 and file_ext.isalnum()): file_ext = None # Basic validation
                                        
                                        if not file_ext and "filename" in att_item and att_item["filename"] and '.' in att_item["filename"]:
                                            potential_ext = att_item["filename"].split(".")[-1].lower()
                                            if 1 < len(potential_ext) < 6 and potential_ext.isalnum():
                                                file_ext = potential_ext
                                                logger.debug(f"Extracted ext '{file_ext}' from filename: {att_item['filename']}")

                                    # Fallback to more robust content_type and refined URL parsing if simple checks failed
                                    if not file_ext and content_type:
                                        logger.debug(f"No ext from initial checks, trying content_type: {content_type}")
                                        if content_type.startswith("image/"): file_ext = content_type.split("/")[-1]
                                        elif content_type == "application/pdf": file_ext = "pdf"
                                        elif content_type.startswith("video/"): file_ext = content_type.split("/")[-1]
                                        elif content_type.startswith("audio/"): file_ext = content_type.split("/")[-1]
                                        if file_ext == "mpeg": file_ext = "mp3"
                                        if file_ext: logger.debug(f"Inferred ext '{file_ext}' from content_type: {content_type}")
                                    
                                    if not file_ext and url: # Refined URL parsing
                                        logger.debug(f"Still no ext, trying more robust URL parsing: {url}")
                                        try:
                                            parsed_url_path = urlparse(url).path
                                            filename_from_url = os.path.basename(parsed_url_path)
                                            if '.' in filename_from_url:
                                                potential_ext = filename_from_url.split(".")[-1].lower()
                                                if 1 < len(potential_ext) < 6 and potential_ext.isalnum():
                                                    file_ext = potential_ext
                                                    logger.debug(f"Extracted ext '{file_ext}' from parsed URL path.")
                                        except Exception as e_url_parse:
                                            logger.warning(f"Could not parse URL for extension: {url}, Error: {e_url_parse}")

                                    if file_ext == "jpeg": file_ext = "jpg" # Normalize

                                    if not file_ext:
                                        logger.warning(f"Still could not determine file extension for att in msg {msg_id}: {original_filename}, {content_type}, {url}")
                                        continue
                                    
                                    logger.debug(f"Final determined ext: '{file_ext}'. Comparing with: {normalized_filter_types or 'All'}")
                                    
                                    if not normalized_filter_types or file_ext in normalized_filter_types:
                                        logger.info(f"Attachment PASSED filter: msg_id={msg_id}, filename='{original_filename}', ext='{file_ext}'. Attempting download.")
                                        try:
                                            if not url:
                                                logger.error(f"Skipping download for att in msg {msg_id}: URL is missing.")
                                                continue
                                                
                                            dl_filename = original_filename
                                            if not dl_filename:
                                                dl_filename = f"attachment_{msg_id}.{file_ext}"
                                            
                                            safe_dl_filename = f"{msg_id}_{dl_filename}"
                                            file_path = self.api.download_attachment(url, safe_dl_filename)
                                            logger.info(f"Successfully downloaded: {file_path}")
                                            
                                            # Track the downloaded file
                                            if msg_id not in self.downloaded_attachment_details:
                                                self.downloaded_attachment_details[msg_id] = []
                                                
                                            self.downloaded_attachment_details[msg_id].append({
                                                "path": file_path, # This is now the correct path from download_attachment
                                                "filename": dl_filename, # original filename part
                                                "content_type": att_item.get("content_type", ""),
                                                "message": msg
                                            })
                                        except Exception as e:
                                            logger.error(f"Error downloading attachment (msg {msg_id}, url {url}): {e}")
                                    else:
                                        logger.info(f"Attachment SKIPPED by filter: msg_id={msg_id}, filename='{original_filename}', ext='{file_ext}'")
                results[channel_id] = messages
            
            all_messages = []
            for channel_id, chan_messages in results.items():
                for msg_item in chan_messages:
                    msg_item["channel_id"] = channel_id
                    all_messages.append(msg_item)
            self.scraped_messages = all_messages
            
            attachment_count = sum(len(v) for v in self.downloaded_attachment_details.values())
            preview_data = self._create_preview_dataframe(all_messages) # Pass all_messages
            
            return (f"[EXTRACTION COMPLETE]\\nMESSAGES: {len(all_messages)}\\nATTACHMENTS: {attachment_count} found\\nCHANNELS: {len(channel_ids)}\\nSTATUS: Ready for download",
                    preview_data, True)
        except Exception as e:
            logger.error(f"Critical error in scrape_messages: {e}", exc_info=True)
            return f"[EXTRACTION FAILED] Error: {str(e)}", None, False
            
    def _clear_temp_files(self):
        """Clear temporary files and directories from previous runs"""
        try:
            temp_dir = tempfile.gettempdir()
            # Clear any discrape-related temp directories and specific zip files
            for item_name in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item_name)
                if item_name.startswith("discrape_"):
                    if os.path.isdir(item_path):
                        try:
                            shutil.rmtree(item_path)
                            print(f"Cleared temp directory: {item_name}")
                        except Exception as e:
                            print(f"Error clearing temp directory {item_name}: {e}")
                    elif os.path.isfile(item_path) and \
                         (item_name.startswith("discrape_extraction_") or \
                          item_name.startswith("discrape_lora_dataset_")) and \
                         item_name.endswith(".zip"):
                        try:
                            os.remove(item_path)
                            print(f"Cleared temp file: {item_name}")
                        except Exception as e:
                            print(f"Error clearing temp file {item_name}: {e}")
        except Exception as e:
            # Added more specific error logging for directory listing
            logger.error(f"Error listing or processing temp directory contents: {e}")
            print(f"Error listing or processing temp directory contents: {e}")
    
    def _create_preview_dataframe(self, messages):
        """Create a preview dataframe from messages"""
        preview_data = []
        for msg in messages[:100]:  # Limit to 100 for preview
            attachment_count = 0
            if "attachments" in msg:
                attachments = msg["attachments"]
                if isinstance(attachments, str):
                    try:
                        # Try to parse JSON string
                        attachments = json.loads(attachments)
                    except:
                        attachments = []
                
                if isinstance(attachments, list):
                    attachment_count = len(attachments)
            
            preview_data.append({
                "Timestamp": msg.get("timestamp", ""),
                "Author": f"{msg.get('author', {}).get('username', 'Unknown')}#{msg.get('author', {}).get('discriminator', '0000')}",
                "Content": msg.get("content", "")[:100] + ("..." if len(msg.get("content", "")) > 100 else ""),
                "Attachments": attachment_count,
                "Channel ID": msg.get("channel_id", "")
            })
            
        return pd.DataFrame(preview_data)
    
    def _sanitize_filename_for_zip(self, filename: str, max_len: int = 150) -> str:
        """Sanitizes and truncates a filename to be safe for ZIP archives and Windows paths."""
        name, ext = os.path.splitext(filename)
        
        # Windows illegal chars: < > : " / \ | ? * and control chars 0-31
        name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', name)
        name = name.strip('. ')

        # Ensure extension is not overly long
        if len(ext) > 10: # Arbitrary limit for extension length
            ext = ext[:10]

        if len(name) + len(ext) > max_len:
            available_len_for_name = max_len - len(ext)
            if available_len_for_name < 1:
                name = "truncated_file"
                # Further ensure this fallback fits
                if len(name) + len(ext) > max_len:
                    current_available_for_name = max_len - len(ext)
                    if current_available_for_name > 0:
                        name = name[:current_available_for_name]
                    else:
                        name = ""
            else:
                name = name[:available_len_for_name]
        
        # Ensure name part is not empty after operations
        if not name and ext: # If name became empty but there is an extension
            name = "file"
        elif not name and not ext: # If both name and extension are empty
            return "default_filename" # Return a fixed default

        return f"{name}{ext}"

    def download_results(self, output_format):
        """Download scraped results with attachments as a ZIP file"""
        try:
            if not self.scraped_messages:
                return "[ERROR] No data available. Run extraction first.", None
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = tempfile.gettempdir()
            export_dir = os.path.join(temp_dir, f"discrape_data_export_{timestamp}") # Temp dir for building ZIP contents
            attachments_export_dir = os.path.join(export_dir, "attachments") # This will be export_dir/attachments/
            os.makedirs(export_dir, exist_ok=True)
            os.makedirs(attachments_export_dir, exist_ok=True)
            
            export_data_for_csv_json = []
            attachment_details_for_index_csv = [] # Simplified list for attachment_index.csv
            
            logger.info(f"Preparing download. Export directory for zipping: {export_dir}")

            for msg in self.scraped_messages:
                msg_id = msg.get("id", "")
                attachments_metadata_for_main_export = []
                if "attachments" in msg and msg["attachments"]:
                    discord_attachments_metadata_list = msg["attachments"]
                    if isinstance(discord_attachments_metadata_list, str):
                        try: discord_attachments_metadata_list = json.loads(discord_attachments_metadata_list)
                        except: discord_attachments_metadata_list = []
                    if not isinstance(discord_attachments_metadata_list, list):
                        discord_attachments_metadata_list = [discord_attachments_metadata_list]
                    
                    for att_meta in discord_attachments_metadata_list:
                        if isinstance(att_meta, dict):
                            attachments_metadata_for_main_export.append({
                                "id": att_meta.get("id", ""),
                                "filename": att_meta.get("filename", ""),
                                "url": att_meta.get("url", ""),
                                "content_type": att_meta.get("content_type", "")
                            })

                if msg_id in self.downloaded_attachment_details:
                    for downloaded_att_detail in self.downloaded_attachment_details[msg_id]:
                        source_file_path_on_disk = downloaded_att_detail.get("path")
                        # original_filename_from_discord is the short name (e.g., "image.png")
                        original_filename_from_discord = downloaded_att_detail.get("filename") 
                        
                        if source_file_path_on_disk and os.path.exists(source_file_path_on_disk) and original_filename_from_discord:
                            # filename_as_downloaded is how it's named on disk (e.g. "{msg_id}_{original_filename_from_discord}")
                            filename_as_downloaded = os.path.basename(source_file_path_on_disk)
                            # Sanitize this potentially long msg_id + original_filename combination
                            sanitized_filename_for_zip = self._sanitize_filename_for_zip(filename_as_downloaded)

                            # Destination will be export_dir/attachments/sanitized_filename_for_zip
                            dest_file_path_in_export_attachments_dir = os.path.join(attachments_export_dir, sanitized_filename_for_zip)
                            
                            try:
                                shutil.copy2(source_file_path_on_disk, dest_file_path_in_export_attachments_dir)
                                logger.info(f"Copied to export structure: {dest_file_path_in_export_attachments_dir}")
                                
                                # Path as it will appear in the ZIP, relative to ZIP root
                                relative_path_in_zip = os.path.join("attachments", sanitized_filename_for_zip).replace('\\', '/')
                                # Use a simpler friendly name for the hyperlink
                                hyperlink_formula = f'=HYPERLINK("{relative_path_in_zip}", "Open Attachment")'
                                
                                attachment_details_for_index_csv.append({
                                    "message_id": msg_id,
                                    "original_filename_from_discord": original_filename_from_discord,
                                    "filename_in_zip": sanitized_filename_for_zip,
                                    "hyperlink_formula": hyperlink_formula,
                                    "relative_path_for_linking": relative_path_in_zip
                                })
                            except Exception as e:
                                logger.error(f"Failed to copy {source_file_path_on_disk} for zipping: {e}")
                        else:
                            logger.warning(f"Source file missing/invalid for msg {msg_id}: {downloaded_att_detail}")
                
                export_data_for_csv_json.append({
                    "id": msg_id, "channel_id": msg.get("channel_id", ""),
                    "timestamp": msg.get("timestamp", ""),
                    "author_id": msg.get("author", {}).get("id", ""),
                    "author_username": msg.get("author", {}).get("username", ""),
                    "author_discriminator": msg.get("author", {}).get("discriminator", ""),
                    "content": msg.get("content", ""),
                    "attachments": json.dumps(attachments_metadata_for_main_export),
                    "mentions": json.dumps([{"id": m.get("id", ""),"username": m.get("username", "")} for m in msg.get("mentions", [])]),
                    "referenced_message_id": msg.get("referenced_message", {}).get("id", "") if msg.get("referenced_message") else ""
                })
                
            df_main_data = pd.DataFrame(export_data_for_csv_json)
            data_file_basename = f"discrape_data_{timestamp}"
            if output_format == "CSV":
                main_data_file_path_in_export = os.path.join(export_dir, f"{data_file_basename}.csv")
                df_main_data.to_csv(main_data_file_path_in_export, index=False, encoding="utf-8-sig")
            else: # JSON
                main_data_file_path_in_export = os.path.join(export_dir, f"{data_file_basename}.json")
                df_main_data.to_json(main_data_file_path_in_export, orient="records", indent=2)
                
            attachment_index_csv_path_in_export = None
            if attachment_details_for_index_csv:
                df_attachment_index = pd.DataFrame(attachment_details_for_index_csv)
                # Select and rename columns for the final attachment_index.csv
                df_attachment_index = df_attachment_index[[
                    "message_id", 
                    "original_filename_from_discord", 
                    "filename_in_zip", 
                    "hyperlink_formula" # This column now holds the Excel formula
                ]]
                df_attachment_index.columns = [
                    "message_id", 
                    "original_filename", 
                    "filename_in_zip", 
                    "link_to_file_in_zip" # Renamed column for clarity
                ]
                attachment_index_csv_path_in_export = os.path.join(export_dir, "attachment_index.csv")
                df_attachment_index.to_csv(attachment_index_csv_path_in_export, index=False, encoding="utf-8-sig")
            
            zip_path_final = os.path.join(temp_dir, f"discrape_extraction_{timestamp}.zip")
            import zipfile
            logger.info(f"Creating ZIP file: {zip_path_final}")
            with zipfile.ZipFile(zip_path_final, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add main data file (CSV or JSON)
                zipf.write(main_data_file_path_in_export, os.path.basename(main_data_file_path_in_export))
                
                # Add attachment_index.csv if it was created
                if attachment_index_csv_path_in_export:
                    zipf.write(attachment_index_csv_path_in_export, os.path.basename(attachment_index_csv_path_in_export))
                
                # Add all files from the attachments_export_dir (which is now flat)
                if os.path.exists(attachments_export_dir):
                    for item_name_in_attachments_dir in os.listdir(attachments_export_dir):
                        item_full_path = os.path.join(attachments_export_dir, item_name_in_attachments_dir)
                        if os.path.isfile(item_full_path):
                            # Arcname will be "attachments/sanitized_filename_for_zip"
                            arcname_in_zip = os.path.join("attachments", item_name_in_attachments_dir).replace('\\', '/')
                            zipf.write(item_full_path, arcname_in_zip)
                            logger.debug(f"Added to ZIP: {item_full_path} as {arcname_in_zip}")
            
            logger.info(f"ZIP file created successfully: {zip_path_final}")
            return f"[DATA READY] Click to download extraction data with attachments", zip_path_final
        except Exception as e:
            logger.error(f"Failed to download results: {e}", exc_info=True)
            return f"[ERROR] Download failed: {str(e)}", None
    
    def scrape_image_prompts(self, channel_id, bot_ids_input, max_messages, progress=gr.Progress()):
        """Scrape image-prompt pairs from a channel"""
        try:
            # Clean up temp files first
            self._clear_temp_files()
            
            if not channel_id:
                return "[ERROR] Channel ID required for extraction", None, False
                
            # Parse bot IDs
            bot_ids = [bid.strip() for bid in bot_ids_input.split(",")]
            
            # Add to recent channels
            self.config_manager.add_recent_channel(channel_id)
            
            # Scrape messages
            progress(0, "[SCANNING] Channel for images...")
            messages = self.scraper.scrape_channel(
                channel_id,
                max_messages,
                lambda current, total: progress(0.5 * current / total, f"[SCANNING] Messages: {current}/{total}")
            )
            
            # Extract image-prompt pairs
            progress(0.5, "[ANALYZING] Extracting image-prompt pairs...")
            pairs = self.scraper.extract_image_prompt_pairs(messages, bot_ids)
            
            # Download images temporarily for display
            progress(0.6, "[PROCESSING] Images for display...")
            image_paths = []
            total_images = sum(len(pair["images"]) for pair in pairs)
            current = 0
            
            # Use temp directory for storing images that will be displayed in the UI
            temp_dir = tempfile.gettempdir()
            temp_image_dir = os.path.join(temp_dir, "discrape_temp_images")
            os.makedirs(temp_image_dir, exist_ok=True)
            
            # Store image data for export later
            image_data_for_export = []
            
            for pair in pairs:
                for image_info in pair["images"]:
                    try:
                        # Create a more descriptive filename
                        timestamp = datetime.fromisoformat(pair["timestamp"].replace("Z", "+00:00")).strftime("%Y%m%d_%H%M%S")
                        clean_prompt = re.sub(r'[^\w\s-]', '', pair.get("prompt", "unknown"))[:30]
                        clean_prompt = clean_prompt.replace(" ", "_")
                        
                        # Download to temp folder for display
                        temp_filename = f"temp_{timestamp}_{pair['message_id']}_{os.path.basename(image_info['filename'])}"
                        temp_path = os.path.join(temp_image_dir, temp_filename)
                        
                        # Download directly to temp
                        response = requests.get(image_info["url"], headers=self.api._get_headers())
                        response.raise_for_status()
                        
                        with open(temp_path, "wb") as f:
                            f.write(response.content)
                        
                        # Add to paths for gallery display
                        image_paths.append((temp_path, pair.get("prompt", "No prompt found")))
                        
                        # Store data for export
                        image_data_for_export.append({
                            "message_id": pair["message_id"],
                            "timestamp": pair["timestamp"],
                            "prompt": pair["prompt"],
                            "content": pair["content"],
                            "image_url": image_info["url"],
                            "image_filename": image_info["filename"],
                            "author_id": pair["author"]["id"],
                            "author_username": pair["author"]["username"]
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing image {image_info['url']}: {e}")
                    
                    current += 1
                    progress(0.6 + 0.4 * current / total_images, f"[PROCESSING] Images: {current}/{total_images}")
            
            # Save for later use
            self.scraped_image_prompts = pairs
            self.image_data_for_export = image_data_for_export
            
            # Prepare gallery
            gallery_items = []
            for path, prompt in image_paths:
                try:
                    # Load image for display
                    gallery_items.append((path, prompt))
                except Exception as e:
                    logger.error(f"Error loading image {path}: {e}")
            
            return (
                f"[EXTRACTION COMPLETE]\n"
                f"IMAGE-PROMPT PAIRS: {len(pairs)}\n"
                f"IMAGES RETRIEVED: {len(gallery_items)}\n"
                f"STATUS: Ready for download",
                gallery_items,
                True  # Show download button
            )
        except Exception as e:
            logger.error(f"Failed to scrape image-prompt pairs: {e}")
            return f"[ERROR] Image extraction failed: {str(e)}", None, False
    
    def download_image_pairs(self):
        """Download image-prompt pairs as a ZIP file with image files and CSV index"""
        try:
            if not hasattr(self, 'image_data_for_export') or not self.image_data_for_export:
                return "[ERROR] No image data. Run extraction first.", None
            
            # Clean up temp files first
            self._clear_temp_files()
            
            # Create a temp directory for our export
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = tempfile.gettempdir()
            export_dir = os.path.join(temp_dir, f"discrape_export_{timestamp}")
            images_dir = os.path.join(export_dir, "images")
            os.makedirs(export_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            
            # Create a list for CSV data with references to image files
            csv_data = []
            
            # Process each image
            total_images = len(self.image_data_for_export)
            downloaded_images = 0
            
            with open(os.path.join(export_dir, "debug_log.txt"), "w") as debug_log:
                debug_log.write(f"Starting image export at {timestamp}\n")
                debug_log.write(f"Total images to download: {total_images}\n\n")
                
                for idx, item in enumerate(self.image_data_for_export):
                    try:
                        debug_log.write(f"Processing image {idx+1}/{total_images}\n")
                        debug_log.write(f"URL: {item['image_url']}\n")
                        
                        # Create clean filename with prompt
                        clean_prompt = re.sub(r'[^\w\s-]', '', item.get('prompt', 'unknown'))[:30]
                        clean_prompt = clean_prompt.replace(" ", "_")
                        safe_filename = f"{item['message_id']}_{clean_prompt}.png"
                        
                        # Build absolute paths
                        image_path = os.path.join(images_dir, safe_filename)
                        
                        # Download directly with fresh headers
                        try:
                            headers = {
                                "Authorization": self.api.config.get_token(),
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                            }
                            
                            response = requests.get(item['image_url'], headers=headers, timeout=30)
                            status_code = response.status_code
                            debug_log.write(f"Download status code: {status_code}\n")
                            
                            if status_code == 200:
                                # Save the image file
                                with open(image_path, 'wb') as f:
                                    f.write(response.content)
                                    
                                file_size = len(response.content)
                                debug_log.write(f"Saved {file_size} bytes to {image_path}\n")
                                downloaded_images += 1
                                
                                # Add data to CSV with reference to the image file
                                csv_data.append({
                                    "message_id": item["message_id"],
                                    "timestamp": item["timestamp"],
                                    "prompt": item.get("prompt", ""),
                                    "filename": safe_filename,
                                    "image_path": os.path.join("images", safe_filename),
                                    "author_id": item.get("author_id", ""),
                                    "author_username": item.get("author_username", "")
                                })
                            else:
                                debug_log.write(f"ERROR: Bad status code when downloading\n")
                                # Still add entry to CSV but without image path
                                csv_data.append({
                                    "message_id": item["message_id"],
                                    "timestamp": item["timestamp"],
                                    "prompt": item.get("prompt", ""),
                                    "filename": "",
                                    "image_path": "",
                                    "author_id": item.get("author_id", ""),
                                    "author_username": item.get("author_username", "")
                                })
                        except Exception as e:
                            debug_log.write(f"ERROR downloading image: {str(e)}\n")
                            
                            # Still add entry to CSV but without image path
                            csv_data.append({
                                "message_id": item["message_id"],
                                "timestamp": item["timestamp"],
                                "prompt": item.get("prompt", ""),
                                "filename": "",
                                "image_path": "",
                                "author_id": item.get("author_id", ""),
                                "author_username": item.get("author_username", "")
                            })
                    except Exception as outer_e:
                        debug_log.write(f"ERROR processing image entry: {str(outer_e)}\n")
                
                debug_log.write(f"\nSummary: Downloaded {downloaded_images}/{total_images} images\n")
            
                # Create and save the CSV with image references
                df = pd.DataFrame(csv_data)
                csv_path = os.path.join(export_dir, "image_prompts.csv")
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                debug_log.write(f"Created CSV at {csv_path}\n")
                
                # Create a simple text file with just the prompts for easy copying
                prompts_txt_path = os.path.join(export_dir, "prompts.txt")
                with open(prompts_txt_path, 'w', encoding='utf-8') as f:
                    for item in csv_data:
                        if item.get("prompt"):
                            f.write(f"{item['prompt']}\n")
                debug_log.write(f"Created prompts.txt at {prompts_txt_path}\n")
                
                # Create ZIP file with everything
                zip_path = os.path.join(temp_dir, f"discrape_lora_dataset_{timestamp}.zip")
                import zipfile
                
                debug_log.write(f"Creating ZIP file at {zip_path}\n")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add debug log
                    zipf.write(os.path.join(export_dir, "debug_log.txt"), "debug_log.txt")
                    
                    # Add CSV
                    zipf.write(csv_path, os.path.basename(csv_path))
                    debug_log.write(f"Added {os.path.basename(csv_path)} to ZIP\n")
                    
                    # Add prompts.txt
                    zipf.write(prompts_txt_path, os.path.basename(prompts_txt_path))
                    debug_log.write(f"Added {os.path.basename(prompts_txt_path)} to ZIP\n")
                    
                    # Add all images
                    for root, _, files in os.walk(images_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, export_dir)
                            zipf.write(file_path, arcname)
                            debug_log.write(f"Added {arcname} to ZIP\n")
                
                # Count files in zip for verification
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    file_count = len(zipf.namelist())
                    debug_log.write(f"ZIP contains {file_count} files\n")
            
            return f"[DATA READY] Click to download LoRA dataset ({downloaded_images} images)", zip_path
        except Exception as e:
            logger.error(f"Failed to download image-prompt pairs: {e}")
            return f"[ERROR] Download failed: {str(e)}", None
    
    def save_settings(self, download_folder, rate_limit):
        """Save settings"""
        try:
            self.config_manager.set_download_folder(download_folder)
            self.config_manager.set_rate_limit(rate_limit)
            return "[CONFIG UPDATED] Settings saved successfully"
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return f"[ERROR] Settings update failed: {str(e)}"

def main():
    """Main entry point"""
    interface = GradioInterface()
    app = interface.build_interface()
    try:
        # Try to safely print ASCII art, falling back to simple message if it fails
        print(DISCRAPE_ASCII)
    except UnicodeEncodeError:
        print("DiScrape v1.0.0 initialized.")
    
    print("\nThe digital extraction begins...\n")
    
    # Get command line arguments (for handling the case when run through the .bat file)
    if len(sys.argv) > 1 and sys.argv[1] == "--share":
        app.launch(share=True, allowed_paths=[os.path.expanduser("~/Downloads/discrape")])
    else:
        app.launch(allowed_paths=[os.path.expanduser("~/Downloads/discrape")])


if __name__ == "__main__":
    main()
