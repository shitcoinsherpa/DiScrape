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
            if "attachments" in msg:
                for attachment in msg["attachments"]:
                    file_ext = attachment["url"].split(".")[-1].lower()
                    if not attachment_types or file_ext in attachment_types:
                        msg_id = msg["id"]
                        if msg_id not in attachment_urls:
                            attachment_urls[msg_id] = []
                        attachment_urls[msg_id].append({
                            "url": attachment["url"],
                            "filename": attachment["filename"],
                            "content_type": attachment.get("content_type", ""),
                            "message": msg
                        })
        
        # Download the attachments
        downloaded_files = {}
        total = sum(len(attachments) for attachments in attachment_urls.values())
        current = 0
        
        for msg_id, attachments in attachment_urls.items():
            downloaded_files[msg_id] = []
            for attachment in attachments:
                try:
                    filename = f"{msg_id}_{attachment['filename']}"
                    file_path = self.api.download_attachment(attachment["url"], filename)
                    downloaded_files[msg_id].append({
                        "path": file_path,
                        "filename": attachment["filename"],
                        "content_type": attachment["content_type"],
                        "message": attachment["message"]
                    })
                except Exception as e:
                    logger.error(f"Error downloading attachment {attachment['url']}: {e}")
                
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
        
        for msg in messages:
            # Check if message is from a bot we're interested in
            if msg.get("author", {}).get("id") in bot_ids and "attachments" in msg:
                # Get image attachments
                images = []
                for attachment in msg["attachments"]:
                    if attachment.get("content_type", "").startswith("image/"):
                        images.append({
                            "url": attachment["url"],
                            "filename": attachment["filename"],
                            "width": attachment.get("width"),
                            "height": attachment.get("height")
                        })
                
                if not images:
                    continue
                    
                # Try to extract prompt from the message content
                content = msg.get("content", "")
                prompt = None
                
                # Look for common patterns in Midjourney messages
                # Pattern 1: "**prompt** - <parameters>"
                match = re.search(r"\*\*(.*?)\*\*", content)
                if match:
                    prompt = match.group(1)
                # Pattern 2: Look for the last message reference which might contain the prompt
                elif msg.get("referenced_message"):
                    prompt = msg.get("referenced_message", {}).get("content", "")
                
                # Add to pairs
                pairs.append({
                    "message_id": msg["id"],
                    "timestamp": msg["timestamp"],
                    "prompt": prompt,
                    "content": content,  # Full content for fallback
                    "images": images,
                    "author": {
                        "id": msg["author"]["id"],
                        "username": msg["author"]["username"],
                        "discriminator": msg["author"].get("discriminator", "0000"),
                        "bot": msg["author"].get("bot", False)
                    }
                })
                
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
                                
                                download_btn = gr.Button(">> DOWNLOAD DATA", variant="primary")
                                
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
                                    maximum=5000,
                                    step=10,
                                    value=1000
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
                            download_image_pairs_btn = gr.Button(">> DOWNLOAD IMAGE DATA", visible=False)
                            # Add file component for image data download
                            image_download_file = gr.File(
                                label="> DOWNLOAD IMAGE DATA",
                                visible=False
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
            # Check if any channels were selected
            if not selected_channels_text or selected_channels_text.strip() == "":
                return "[ERROR] No extraction targets selected", None, False
                
            # Parse the comma-separated channel IDs
            channel_ids = [c.strip() for c in selected_channels_text.split(",")]
            
            if not channel_ids:
                return "[ERROR] No extraction targets selected", None, False
            
            # Add to recent channels
            for channel_id in channel_ids:
                self.config_manager.add_recent_channel(channel_id)
                
            # Scrape messages with progress tracking
            results = {}
            total_channels = len(channel_ids)
            
            for i, channel_id in enumerate(channel_ids):
                progress(i / total_channels, f"[EXTRACTING] Channel {i+1}/{total_channels}")
                
                # Get messages
                messages = self.scraper.scrape_channel(
                    channel_id,
                    max_messages,
                    lambda current, total: progress((i + current/total) / total_channels, 
                                                  f"[EXTRACTING] Channel {i+1}/{total_channels}: {current}/{total} packets")
                )
                
                results[channel_id] = messages
                
            # Flatten messages for display and export
            all_messages = []
            for channel_id, messages in results.items():
                for msg in messages:
                    # Add channel_id to each message
                    msg["channel_id"] = channel_id
                    all_messages.append(msg)
                    
            self.scraped_messages = all_messages
            
            # Download attachments if requested
            if include_attachments:
                progress(0, "[DOWNLOADING] Attachments...")
                self.scraper.download_attachments(
                    all_messages,
                    attachment_types,
                    lambda current, total: progress(current / total, f"[DOWNLOADING] Attachments: {current}/{total}")
                )
                
            # Create preview dataframe
            preview_data = self._create_preview_dataframe(all_messages)
            
            # Return whether to show download button - always true for visibility
            return (
                f"[EXTRACTION COMPLETE]\n"
                f"MESSAGES: {len(all_messages)}\n"
                f"CHANNELS: {len(channel_ids)}\n"
                f"STATUS: Ready for download",
                preview_data,
                True  # Always show download button
            )
        except Exception as e:
            logger.error(f"Failed to scrape messages: {e}")
            return f"[EXTRACTION FAILED] Error: {str(e)}", None, False
    
    def _create_preview_dataframe(self, messages):
        """Create a preview dataframe from messages"""
        preview_data = []
        for msg in messages[:100]:  # Limit to 100 for preview
            preview_data.append({
                "Timestamp": msg.get("timestamp", ""),
                "Author": f"{msg.get('author', {}).get('username', 'Unknown')}#{msg.get('author', {}).get('discriminator', '0000')}",
                "Content": msg.get("content", "")[:100] + ("..." if len(msg.get("content", "")) > 100 else ""),
                "Attachments": len(msg.get("attachments", [])),
                "Channel ID": msg.get("channel_id", "")
            })
            
        return pd.DataFrame(preview_data)
    
    def download_results(self, output_format):
        """Download scraped results in selected format"""
        try:
            if not self.scraped_messages:
                return "[ERROR] No data available. Run extraction first.", None
                
            # Prepare data for export
            export_data = []
            for msg in self.scraped_messages:
                export_data.append({
                    "id": msg.get("id", ""),
                    "channel_id": msg.get("channel_id", ""),
                    "timestamp": msg.get("timestamp", ""),
                    "author_id": msg.get("author", {}).get("id", ""),
                    "author_username": msg.get("author", {}).get("username", ""),
                    "author_discriminator": msg.get("author", {}).get("discriminator", ""),
                    "content": msg.get("content", ""),
                    "attachments": json.dumps([{
                        "id": att.get("id", ""),
                        "filename": att.get("filename", ""),
                        "url": att.get("url", ""),
                        "content_type": att.get("content_type", "")
                    } for att in msg.get("attachments", [])]),
                    "mentions": json.dumps([{
                        "id": mention.get("id", ""),
                        "username": mention.get("username", "")
                    } for mention in msg.get("mentions", [])]),
                    "referenced_message_id": msg.get("referenced_message", {}).get("id", "") if msg.get("referenced_message") else ""
                })
                
            # Create dataframe
            df = pd.DataFrame(export_data)
            
            # Save to temporary file for download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = tempfile.gettempdir()
            
            if output_format == "CSV":
                filename = os.path.join(temp_dir, f"discrape_data_{timestamp}.csv")
                df.to_csv(filename, index=False, encoding="utf-8-sig")
            elif output_format == "Excel":
                # Avoid using to_excel which requires openpyxl
                filename = os.path.join(temp_dir, f"discrape_data_{timestamp}.csv")
                df.to_csv(filename, index=False, encoding="utf-8-sig")
                return f"[NOTE] Excel format requires openpyxl module. Saved as CSV instead.", filename
            else:  # JSON
                filename = os.path.join(temp_dir, f"discrape_data_{timestamp}.json")
                df.to_json(filename, orient="records", indent=2)
                
            return f"[DATA READY] Click to download file", filename
        except Exception as e:
            logger.error(f"Failed to download results: {e}")
            return f"[ERROR] Download failed: {str(e)}", None
    
    def scrape_image_prompts(self, channel_id, bot_ids_input, max_messages, progress=gr.Progress()):
        """Scrape image-prompt pairs from a channel"""
        try:
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
            
            # Download images
            progress(0.6, "[DOWNLOADING] Images...")
            image_paths = []
            total_images = sum(len(pair["images"]) for pair in pairs)
            current = 0
            
            for pair in pairs:
                for image_info in pair["images"]:
                    try:
                        # Create a more descriptive filename
                        timestamp = datetime.fromisoformat(pair["timestamp"].replace("Z", "+00:00")).strftime("%Y%m%d_%H%M%S")
                        clean_prompt = re.sub(r'[^\w\s-]', '', pair.get("prompt", "unknown"))[:30]
                        clean_prompt = clean_prompt.replace(" ", "_")
                        
                        filename = f"{timestamp}_{pair['message_id']}_{clean_prompt}_{image_info['filename']}"
                        file_path = self.api.download_attachment(image_info["url"], f"images/{filename}")
                        
                        # Add to image_paths for gallery display
                        image_paths.append((file_path, pair.get("prompt", "No prompt found")))
                    except Exception as e:
                        logger.error(f"Error downloading image {image_info['url']}: {e}")
                    
                    current += 1
                    progress(0.6 + 0.4 * current / total_images, f"[DOWNLOADING] Images: {current}/{total_images}")
            
            # Save for later use
            self.scraped_image_prompts = pairs
            
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
        """Download image-prompt pairs data"""
        try:
            if not self.scraped_image_prompts:
                return "[ERROR] No image data. Run extraction first.", None
                
            # Create CSV with image-prompt data
            export_data = []
            for pair in self.scraped_image_prompts:
                for image in pair["images"]:
                    export_data.append({
                        "message_id": pair["message_id"],
                        "timestamp": pair["timestamp"],
                        "prompt": pair.get("prompt", ""),
                        "content": pair["content"],
                        "image_url": image["url"],
                        "image_filename": image["filename"],
                        "author_id": pair["author"]["id"],
                        "author_username": pair["author"]["username"],
                        "local_path": f"images/{pair['message_id']}_{image['filename']}"
                    })
            
            # Create dataframe
            df = pd.DataFrame(export_data)
            
            # Save to temporary file for download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = tempfile.gettempdir()
            filename = os.path.join(temp_dir, f"discrape_images_{timestamp}.csv")
            
            try:
                df.to_csv(filename, index=False, encoding="utf-8-sig")
            except Exception as e:
                logger.error(f"CSV export failed, using simpler format: {e}")
                # Fallback to simpler export if encoding issues
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("message_id,timestamp,prompt,image_url,image_filename\n")
                    for item in export_data:
                        f.write(f"{item['message_id']},{item['timestamp']},{item['prompt']},{item['image_url']},{item['image_filename']}\n")
            
            return f"[DATA READY] Click to download file", filename
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
        app.launch(share=True)
    else:
        app.launch()

if __name__ == "__main__":
    main()
