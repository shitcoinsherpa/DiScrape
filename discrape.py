import gradio as gr
import requests
import json
import os
import pandas as pd
from datetime import datetime, timedelta
import cryptography.fernet as fernet
import base64
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any
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
import threading
from collections import defaultdict, deque

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
> Infiltrate. Extract. Analyze.        v2.1.0 <
"""

# Matrix-like symbols for animations
MATRIX_CHARS = "!@#$%^&*()_+-=[]{}|;:,./<>?~`abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Known AI bot IDs
KNOWN_AI_BOTS = {
    "936929561302675456": "Midjourney Bot",
    "1022952195194359889": "Midjourney Bot",  # Alternative ID
    "989287882680983572": "BlueWillow",
    "1050422060342321214": "Niji Journey Bot",
    # Add more bot IDs as discovered
}

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
            "channel_cache": {},
            "server_cache": {},
            "known_bots": KNOWN_AI_BOTS,
            "rate_limit": 1.0,
            "download_folder": os.path.expanduser("~/Downloads/discrape"),
            "theme": "dark",
            "auto_retry": True,
            "max_retries": 3,
            "scrape_stats": {
                "total_messages": 0,
                "total_images": 0,
                "total_servers": 0
            }
        }
    
    def save_config(self):
        """Save current configuration to file"""
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2)
            
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
        # Ensure recent_servers exists
        if "recent_servers" not in self.config:
            self.config["recent_servers"] = []
        
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
        
        # Cache server name
        if server_name:
            if "server_cache" not in self.config:
                self.config["server_cache"] = {}
            self.config["server_cache"][server_id] = server_name
            
        self.save_config()
        
    def add_recent_channel(self, channel_id: str, channel_name: str = None):
        """Add a channel to recent channels list"""
        # Ensure recent_channels exists
        if "recent_channels" not in self.config:
            self.config["recent_channels"] = []
            
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
        
        # Cache channel name
        if channel_name:
            if "channel_cache" not in self.config:
                self.config["channel_cache"] = {}
            self.config["channel_cache"][channel_id] = channel_name
            
        self.save_config()
    
    def get_known_bots(self) -> Dict[str, str]:
        """Get known AI bot IDs and names"""
        if "known_bots" not in self.config:
            self.config["known_bots"] = KNOWN_AI_BOTS
            self.save_config()
        return self.config["known_bots"]
    
    def add_known_bot(self, bot_id: str, bot_name: str):
        """Add a new bot to known bots"""
        if "known_bots" not in self.config:
            self.config["known_bots"] = KNOWN_AI_BOTS
        self.config["known_bots"][bot_id] = bot_name
        self.save_config()
    
    def get_channel_name(self, channel_id: str) -> str:
        """Get cached channel name or return ID"""
        if "channel_cache" not in self.config:
            self.config["channel_cache"] = {}
        return self.config["channel_cache"].get(channel_id, channel_id)
    
    def get_server_name(self, server_id: str) -> str:
        """Get cached server name or return ID"""
        if "server_cache" not in self.config:
            self.config["server_cache"] = {}
        return self.config["server_cache"].get(server_id, server_id)
    
    def update_stats(self, messages: int = 0, images: int = 0, servers: int = 0):
        """Update scraping statistics"""
        if "scrape_stats" not in self.config:
            self.config["scrape_stats"] = {"total_messages": 0, "total_images": 0, "total_servers": 0}
        
        self.config["scrape_stats"]["total_messages"] += messages
        self.config["scrape_stats"]["total_images"] += images
        self.config["scrape_stats"]["total_servers"] += servers
        self.save_config()
    
    def get_stats(self) -> Dict[str, int]:
        """Get scraping statistics"""
        if "scrape_stats" not in self.config:
            self.config["scrape_stats"] = {"total_messages": 0, "total_images": 0, "total_servers": 0}
            self.save_config()
        return self.config["scrape_stats"]
    
    def get_download_folder(self) -> str:
        """Get the download folder, creating it if it doesn't exist"""
        folder = self.config.get("download_folder", os.path.expanduser("~/Downloads/discrape"))
        # Normalize path to use OS-appropriate separators
        folder = os.path.normpath(folder)
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
        self.session = requests.Session()
        self._setup_session()
        
    def _setup_session(self):
        """Setup session with retry adapter"""
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry
        
        retry = Retry(
            total=5,  # Increased retries for DNS issues
            backoff_factor=2,  # Longer backoff
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "TRACE"],
            raise_on_status=False,
            connect=5,  # Add connect retries for DNS failures
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
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
        response = self.session.get(f"{self.base_url}/users/@me", headers=self._get_headers())
        if self._handle_rate_limit(response):
            return self.get_current_user()
        
        response.raise_for_status()
        return response.json()
    
    def get_server_info(self, server_id: str) -> Dict:
        """Get server information"""
        response = self.session.get(f"{self.base_url}/guilds/{server_id}", headers=self._get_headers())
        if self._handle_rate_limit(response):
            return self.get_server_info(server_id)
        
        response.raise_for_status()
        return response.json()
    
    def get_server_channels(self, server_id: str) -> List[Dict]:
        """Get all channels in a server"""
        response = self.session.get(f"{self.base_url}/guilds/{server_id}/channels", headers=self._get_headers())
        if self._handle_rate_limit(response):
            return self.get_server_channels(server_id)
        
        response.raise_for_status()
        return response.json()
    
    def get_channel_info(self, channel_id: str) -> Dict:
        """Get channel information"""
        response = self.session.get(f"{self.base_url}/channels/{channel_id}", headers=self._get_headers())
        if self._handle_rate_limit(response):
            return self.get_channel_info(channel_id)
        
        response.raise_for_status()
        return response.json()
    
    def get_channel_messages(self, channel_id: str, limit: int = 100, before: Optional[str] = None, after: Optional[str] = None) -> List[Dict]:
        """Get messages from a channel"""
        params = {"limit": min(limit, 100)}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
            
        response = self.session.get(
            f"{self.base_url}/channels/{channel_id}/messages", 
            headers=self._get_headers(),
            params=params
        )
        
        if self._handle_rate_limit(response):
            return self.get_channel_messages(channel_id, limit, before, after)
        
        response.raise_for_status()
        # Apply rate limit
        time.sleep(self.config.get_rate_limit())
        return response.json()
    
    def download_attachment(self, url: str, filename: str, folder: str = None) -> str:
        """Download attachment from URL"""
        if folder is None:
            folder = self.config.get_download_folder()
        
        # Normalize folder path to use OS-appropriate separators
        folder = os.path.normpath(folder)
        
        # Ensure filename doesn't contain invalid path characters
        filename = re.sub(r'[<>:"|?*]', '_', filename)
        
        save_path = os.path.join(folder, filename)
        save_path = os.path.normpath(save_path)
        
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        response = self.session.get(url, headers=self._get_headers(), stream=True)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return save_path

class DiscordScraper:
    """Main scraper logic"""
    
    def __init__(self, api: DiscordAPI, config: ConfigManager):
        self.api = api
        self.config = config
        self._stop_event = threading.Event()
        
    def stop_scraping(self):
        """Stop the current scraping operation"""
        self._stop_event.set()
        
    def scrape_channel(self, channel_id: str, max_messages: int = 1000, 
                      progress_callback=None, date_after: datetime = None, 
                      date_before: datetime = None, process_immediately=None) -> List[Dict]:
        """Scrape messages from a channel with date filtering"""
        all_messages = []
        before = None
        batch_size = min(100, max_messages)
        self._stop_event.clear()
        
        # Get channel info for name caching
        try:
            channel_info = self.api.get_channel_info(channel_id)
            channel_name = channel_info.get("name", channel_id)
            self.config.add_recent_channel(channel_id, channel_name)
        except:
            channel_name = channel_id
        
        while len(all_messages) < max_messages and not self._stop_event.is_set():
            try:
                messages = self.api.get_channel_messages(channel_id, batch_size, before)
                if not messages:
                    break
                
                # Filter by date if specified
                filtered_messages = []
                for msg in messages:
                    msg_time = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
                    
                    if date_after and msg_time < date_after:
                        # Messages are in reverse chronological order, so we can stop here
                        break
                        
                    if date_before and msg_time > date_before:
                        continue
                        
                    filtered_messages.append(msg)
                
                if not filtered_messages:
                    break
                    
                # Process messages immediately if callback provided
                if process_immediately:
                    for msg in filtered_messages:
                        msg["channel_id"] = channel_id
                        msg["channel_name"] = self.config.get_channel_name(channel_id)
                        process_immediately(msg)
                
                all_messages.extend(filtered_messages)
                
                # Update progress if callback provided
                if progress_callback:
                    progress_callback(len(all_messages), max_messages)
                
                # Set before to the ID of the last message for pagination
                before = messages[-1]["id"]
                
                # If we got fewer messages than requested, we've reached the end
                if len(messages) < batch_size:
                    break
                    
                # Check if we've gone past our date range
                if date_after:
                    last_msg_time = datetime.fromisoformat(messages[-1]["timestamp"].replace("Z", "+00:00"))
                    if last_msg_time < date_after:
                        break
                        
            except Exception as e:
                logger.error(f"Error scraping channel {channel_id}: {e}")
                break
                
        return all_messages[:max_messages]
    
    def scrape_server(self, server_id: str, max_messages_per_channel: int = 1000, 
                     channel_filter: List[str] = None, progress_callback=None,
                     date_after: datetime = None, date_before: datetime = None) -> Dict[str, List[Dict]]:
        """Scrape all or selected channels from a server"""
        results = {}
        self._stop_event.clear()
        
        try:
            # Get all channels
            channels = self.api.get_server_channels(server_id)
            
            # Filter for text channels
            text_channels = [c for c in channels if c["type"] == 0]
            
            # Apply channel filter if specified
            if channel_filter:
                text_channels = [c for c in text_channels if c["id"] in channel_filter]
            
            total_channels = len(text_channels)
            
            for i, channel in enumerate(text_channels):
                if self._stop_event.is_set():
                    break
                    
                channel_id = channel["id"]
                channel_name = channel["name"]
                
                if progress_callback:
                    progress_callback(i, total_channels, channel_name, 0, 0)
                
                # Cache channel name
                self.config.add_recent_channel(channel_id, channel_name)
                
                messages = self.scrape_channel(
                    channel_id, 
                    max_messages_per_channel,
                    lambda msg_curr, msg_total: 
                        progress_callback(i, total_channels, channel_name, msg_curr, msg_total) if progress_callback else None,
                    date_after,
                    date_before
                )
                
                if messages:
                    results[channel_id] = messages
                    
        except Exception as e:
            logger.error(f"Error scraping server {server_id}: {e}")
            
        return results
    
    def download_attachments(self, messages: List[Dict], attachment_types: List[str] = None, 
                           progress_callback=None, organize_by_channel: bool = True) -> Dict[str, List[Dict]]:
        """Download attachments from messages with better organization"""
        if attachment_types is None:
            attachment_types = ["png", "jpg", "jpeg", "gif", "mp4", "webm", "mp3", "wav", "pdf"]
            
        downloaded_files = defaultdict(list)
        attachment_queue = []
        
        # Build queue of attachments to download
        for msg in messages:
            if "attachments" in msg and msg["attachments"]:
                attachments = msg["attachments"]
                if isinstance(attachments, str):
                    try:
                        attachments = json.loads(attachments)
                    except:
                        attachments = []
                
                if not isinstance(attachments, list):
                    attachments = [attachments]
                
                for attachment in attachments:
                    if not isinstance(attachment, dict):
                        continue
                        
                    # Determine file extension
                    file_ext = None
                    if "url" in attachment and attachment["url"]:
                        file_ext = self._extract_file_extension(attachment["url"])
                    if not file_ext and "filename" in attachment and attachment["filename"]:
                        file_ext = attachment["filename"].split(".")[-1].lower()
                    
                    if file_ext and (not attachment_types or file_ext in attachment_types):
                        attachment_queue.append({
                            "message": msg,
                            "attachment": attachment,
                            "file_ext": file_ext
                        })
        
        # Download attachments
        total = len(attachment_queue)
        for i, item in enumerate(attachment_queue):
            if self._stop_event.is_set():
                break
                
            try:
                msg = item["message"]
                attachment = item["attachment"]
                
                url = attachment.get("url", "")
                if not url:
                    continue
                    
                # Create organized folder structure
                if organize_by_channel:
                    channel_id = msg.get("channel_id", "unknown")
                    channel_name = self.config.get_channel_name(channel_id)
                    safe_channel_name = re.sub(r'[^\w\s-]', '', channel_name)[:50]
                    subfolder = os.path.join(self.config.get_download_folder(), f"{channel_id}_{safe_channel_name}")
                else:
                    subfolder = self.config.get_download_folder()
                
                # Normalize subfolder path
                subfolder = os.path.normpath(subfolder)
                os.makedirs(subfolder, exist_ok=True)
                
                # Create filename with timestamp
                timestamp = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00")).strftime("%Y%m%d_%H%M%S")
                original_filename = attachment.get("filename", f"attachment.{item['file_ext']}")
                # Sanitize filename
                safe_filename = re.sub(r'[<>:"|?*]', '_', f"{timestamp}_{msg['id']}_{original_filename}")
                
                # Limit filename length for Windows compatibility (260 char path limit)
                # Reserve space for path: ~100 chars for folder path, leave 150 for filename
                if len(safe_filename) > 150:
                    # Keep timestamp, message ID, and extension
                    name_parts = safe_filename.split('.')
                    extension = name_parts[-1] if len(name_parts) > 1 else ''
                    base_name = '.'.join(name_parts[:-1]) if len(name_parts) > 1 else safe_filename
                    
                    # Truncate the base name while keeping important parts
                    max_base_length = 150 - len(extension) - 1  # -1 for the dot
                    if len(base_name) > max_base_length:
                        # Keep the timestamp and message ID prefix (about 50 chars)
                        prefix = f"{timestamp}_{msg['id']}_"
                        remaining_length = max_base_length - len(prefix)
                        truncated_original = original_filename[:remaining_length] if remaining_length > 0 else ""
                        base_name = prefix + truncated_original
                    
                    safe_filename = f"{base_name}.{extension}" if extension else base_name
                
                # Download file
                file_path = self.api.download_attachment(url, safe_filename, subfolder)
                
                downloaded_files[msg["id"]].append({
                    "path": file_path,
                    "filename": original_filename,
                    "content_type": attachment.get("content_type", ""),
                    "message": msg,
                    "channel_id": msg.get("channel_id"),
                    "channel_name": self.config.get_channel_name(msg.get("channel_id", ""))
                })
                
                if progress_callback:
                    progress_callback(i + 1, total)
                    
            except Exception as e:
                logger.error(f"Error downloading attachment: {e}")
                
        return dict(downloaded_files)
    
    def extract_image_prompt_pairs_advanced(self, messages: List[Dict], bot_ids: List[str] = None, 
                                           download_immediately: bool = False, export_dir: str = None,
                                           progress_callback=None) -> List[Dict]:
        """Extract image-prompt pairs using proper Discord AI bot flow"""
        if bot_ids is None:
            bot_ids = list(self.config.get_known_bots().keys())
        
        pairs = []
        
        # Sort messages by timestamp to ensure chronological order
        sorted_messages = sorted(messages, key=lambda x: x.get("timestamp", ""))
        
        # Create a lookup for messages by ID
        message_lookup = {msg["id"]: msg for msg in sorted_messages}
        
        # Track recent user prompts (last 50 messages per channel)
        channel_user_prompts = defaultdict(lambda: deque(maxlen=50))
        
        for msg in sorted_messages:
            channel_id = msg.get("channel_id", "unknown")
            author_id = msg.get("author", {}).get("id", "")
            is_bot = msg.get("author", {}).get("bot", False)
            
            # Track user messages that could be prompts
            if not is_bot and msg.get("content"):
                channel_user_prompts[channel_id].append({
                    "message": msg,
                    "content": msg["content"],
                    "author_id": author_id,
                    "timestamp": msg["timestamp"]
                })
            
            # Check if this is a bot message with images
            if author_id in bot_ids and msg.get("attachments"):
                # Extract images from bot message
                images = []
                attachments = msg["attachments"]
                if isinstance(attachments, str):
                    try:
                        attachments = json.loads(attachments)
                    except:
                        attachments = []
                
                if not isinstance(attachments, list):
                    attachments = [attachments]
                
                for attachment in attachments:
                    if isinstance(attachment, dict):
                        content_type = attachment.get("content_type", "")
                        if content_type.startswith("image/"):
                            images.append({
                                "url": attachment.get("url"),
                                "filename": attachment.get("filename"),
                                "width": attachment.get("width"),
                                "height": attachment.get("height"),
                                "size": attachment.get("size")
                            })
                
                if not images:
                    continue
                
                # Now find the associated prompt
                prompt = None
                prompt_message = None
                prompt_type = "unknown"
                parameters = {}
                
                # Strategy 1: Check if bot message references a user message
                if msg.get("referenced_message"):
                    ref_msg_id = msg["referenced_message"].get("id") if isinstance(msg["referenced_message"], dict) else msg["referenced_message"]
                    if ref_msg_id in message_lookup:
                        prompt_message = message_lookup[ref_msg_id]
                        prompt = prompt_message.get("content", "")
                        prompt_type = "direct_reference"
                
                # Strategy 2: Check message content for variations/upscales
                bot_content = msg.get("content", "")
                variation_match = re.search(r"(Variations?|Upscaled?|Image #\d+)\s*(?:\(Strong\))?\s*by\s*<@(\d+)>", bot_content)
                if variation_match and not prompt:
                    user_id = variation_match.group(2)
                    variation_type = variation_match.group(1)
                    
                    # Look for the most recent prompt from this user in this channel
                    for user_msg in reversed(list(channel_user_prompts[channel_id])):
                        if user_msg["author_id"] == user_id:
                            prompt = user_msg["content"]
                            prompt_message = user_msg["message"]
                            prompt_type = f"variation_{variation_type.lower()}"
                            break
                
                # Strategy 3: Extract prompt from bot message content (for some bots that echo the prompt)
                if not prompt and bot_content:
                    # First, clean the bot content to remove Discord formatting artifacts
                    # Remove any lines that look like Discord UI elements
                    cleaned_bot_content = bot_content
                    
                    # Remove lines that contain "BOT" tag or timestamp patterns
                    lines = cleaned_bot_content.split('\n')
                    filtered_lines = []
                    for line in lines:
                        # Skip lines that look like Discord UI
                        if any(pattern in line for pattern in ['BOT', 'Today at', 'Yesterday at', ' AM', ' PM']):
                            continue
                        # Skip lines that are just whitespace
                        if not line.strip():
                            continue
                        filtered_lines.append(line)
                    
                    cleaned_bot_content = '\n'.join(filtered_lines)
                    
                    # Look for prompt patterns in bot's message
                    prompt_patterns = [
                        r'prompt:\s*"([^"]+)"',
                        r'prompt:\s*(.+?)(?:\s*--|\s*$)',
                        r'"([^"]+)"\s*--',
                        r'`([^`]+)`',  # Some bots use backticks
                    ]
                    
                    for pattern in prompt_patterns:
                        match = re.search(pattern, cleaned_bot_content, re.IGNORECASE)
                        if match:
                            extracted = match.group(1).strip()
                            # Additional validation - make sure it's not Discord UI text
                            if not any(ui_text in extracted for ui_text in ['BOT', 'Today at', 'Yesterday at']):
                                prompt = extracted
                                prompt_type = "bot_content_extraction"
                                break
                
                # Strategy 4: Find the most recent user message in the channel (within 10 messages)
                if not prompt:
                    recent_prompts = list(channel_user_prompts[channel_id])[-10:]
                    if recent_prompts:
                        # Use the most recent prompt
                        prompt_data = recent_prompts[-1]
                        candidate_prompt = prompt_data["content"]
                        
                        # Validate that it's not Discord UI text or spam
                        if candidate_prompt and len(candidate_prompt) > 3:
                            # Check for common spam/UI patterns
                            spam_patterns = [
                                r'http[s]?://discord\.gg/',  # Discord invites
                                r'@everyone',  # Mass mentions
                                r'free\s+nudes',  # Spam
                                r'^\s*BOT\s*$',  # Just "BOT"
                                r'Today at \d+:\d+',  # Timestamps
                                r'Yesterday at \d+:\d+',
                            ]
                            
                            is_spam = any(re.search(pattern, candidate_prompt, re.IGNORECASE) for pattern in spam_patterns)
                            
                            if not is_spam:
                                prompt = candidate_prompt
                                prompt_message = prompt_data["message"]
                                prompt_type = "recent_user_message"
                
                # Extract parameters from prompt
                if prompt:
                    # Midjourney style parameters
                    param_patterns = [
                        (r'--ar\s+(\d+:\d+)', 'aspect_ratio'),
                        (r'--v\s+(\d+(?:\.\d+)?)', 'version'),
                        (r'--s\s+(\d+)', 'stylize'),
                        (r'--q\s+(\d+(?:\.\d+)?)', 'quality'),
                        (r'--chaos\s+(\d+)', 'chaos'),
                        (r'--no\s+([^\s]+)', 'negative'),
                        (r'--seed\s+(\d+)', 'seed'),
                        (r'--tile', 'tile'),
                        (r'--test', 'test'),
                        (r'--testp', 'testp'),
                        (r'--uplight', 'uplight'),
                        (r'--stop\s+(\d+)', 'stop'),
                        (r'--style\s+(\w+)', 'style'),
                    ]
                    
                    for pattern, param_name in param_patterns:
                        match = re.search(pattern, prompt)
                        if match:
                            if len(match.groups()) > 0:
                                parameters[param_name] = match.group(1)
                            else:
                                parameters[param_name] = True
                
                # Final cleanup of extracted prompt
                if prompt:
                    # Remove any remaining Discord UI artifacts
                    prompt_lines = prompt.split('\n')
                    cleaned_lines = []
                    
                    for line in prompt_lines:
                        # Skip lines that are clearly Discord UI
                        line_lower = line.lower()
                        if any(ui in line_lower for ui in ['bot\n', 'today at', 'yesterday at', '@everyone', 'http://discord.gg', 'https://discord.gg']):
                            continue
                        # Skip very short lines that might be artifacts
                        if len(line.strip()) < 3 and line.strip() not in ['2D', '3D', 'HD', '4K', '8K']:
                            continue
                        cleaned_lines.append(line)
                    
                    # Join cleaned lines
                    prompt = ' '.join(cleaned_lines)
                    # Final cleanup - remove extra whitespace
                    prompt = re.sub(r'\s+', ' ', prompt).strip()
                
                # Clean prompt by removing parameters
                clean_prompt = prompt
                if prompt:
                    # Remove all parameter patterns
                    clean_prompt = re.sub(r'--\w+(?:\s+[^\s-]+)?', '', prompt).strip()
                
                pair_data = {
                    "message_id": msg["id"],
                    "timestamp": msg["timestamp"],
                    "prompt": clean_prompt,
                    "raw_prompt": prompt,
                    "prompt_type": prompt_type,
                    "prompt_message_id": prompt_message["id"] if prompt_message else None,
                    "parameters": parameters,
                    "bot_content": bot_content,
                    "images": images,
                    "author": {
                        "id": msg.get("author", {}).get("id"),
                        "username": msg.get("author", {}).get("username"),
                        "bot": True,
                        "bot_name": self.config.get_known_bots().get(author_id, "Unknown Bot")
                    },
                    "prompt_author": {
                        "id": prompt_message.get("author", {}).get("id") if prompt_message else None,
                        "username": prompt_message.get("author", {}).get("username") if prompt_message else None
                    } if prompt_message else None,
                    "channel_id": channel_id,
                    "channel_name": self.config.get_channel_name(channel_id),
                    "is_variation": "variation" in prompt_type,
                    "is_upscale": "upscale" in prompt_type.lower()
                }
                
                # Download images immediately if requested
                if download_immediately and export_dir and images:
                    images_dir = os.path.join(export_dir, "images")
                    metadata_dir = os.path.join(export_dir, "metadata")
                    os.makedirs(images_dir, exist_ok=True)
                    os.makedirs(metadata_dir, exist_ok=True)
                    
                    # Get total pairs processed so far from checkpoint
                    checkpoint_file = os.path.join(export_dir, "checkpoint.json")
                    pair_counter = 1
                    if os.path.exists(checkpoint_file):
                        try:
                            with open(checkpoint_file, 'r') as f:
                                checkpoint = json.load(f)
                                pair_counter = checkpoint.get("total_pairs", 0) + 1
                        except:
                            pass
                    
                    downloaded_images = []
                    for img_idx, image in enumerate(images):
                        if not image.get("url"):
                            continue
                            
                        # Create filename using pair counter
                        safe_channel = re.sub(r'[^\w\s-]', '', self.config.get_channel_name(channel_id))
                        safe_channel = re.sub(r'\s+', ' ', safe_channel).strip()[:30].replace(' ', '_')
                        
                        if clean_prompt:
                            clean_prompt_fn = re.sub(r'[^\w\s\-_.]', '', clean_prompt)
                            clean_prompt_fn = re.sub(r'\s+', ' ', clean_prompt_fn).strip()[:40].replace(' ', '_')
                        else:
                            clean_prompt_fn = "no_prompt"
                        
                        image_filename = f"{pair_counter:05d}_{img_idx:02d}_{safe_channel}_{clean_prompt_fn}.png"
                        # Ensure filename isn't too long
                        if len(image_filename) > 150:
                            image_filename = f"{pair_counter:05d}_{img_idx:02d}_{safe_channel[:50]}.png"
                        
                        image_path = os.path.join(images_dir, image_filename)
                        
                        # Check if already downloaded
                        if os.path.exists(image_path) and os.path.getsize(image_path) > 1024:
                            logger.info(f"Image already exists: {image_filename}")
                            image["local_filename"] = image_filename
                            image["downloaded"] = True
                            downloaded_images.append(image)
                            continue
                        
                        # Download with retries
                        success = self._download_single_image(image["url"], image_path)
                        
                        if success:
                            image["local_filename"] = image_filename
                            image["downloaded"] = True
                            downloaded_images.append(image)
                            
                            # Create caption file
                            if clean_prompt:
                                caption_path = os.path.join(images_dir, image_filename.replace('.png', '.txt'))
                                with open(caption_path, 'w', encoding='utf-8') as f:
                                    f.write(clean_prompt)
                            
                            # Report progress
                            if progress_callback:
                                progress_callback(f"Downloaded: {image_filename}")
                        else:
                            image["downloaded"] = False
                            logger.error(f"Failed to download image from {image['url']}")
                    
                    # Save metadata for this pair immediately
                    dataset_file = os.path.join(metadata_dir, "dataset.jsonl")
                    with open(dataset_file, 'a', encoding='utf-8') as f:
                        pair_record = {
                            "pair_id": pair_counter,
                            "message_id": msg["id"],
                            "timestamp": msg["timestamp"],
                            "prompt": clean_prompt,
                            "raw_prompt": prompt,
                            "parameters": parameters,
                            "channel_id": channel_id,
                            "channel_name": self.config.get_channel_name(channel_id),
                            "bot_name": self.config.get_known_bots().get(author_id, "Unknown Bot"),
                            "images": downloaded_images,
                            "is_variation": "variation" in prompt_type,
                            "is_upscale": "upscale" in prompt_type.lower()
                        }
                        f.write(json.dumps(pair_record) + '\n')
                    
                    # Update checkpoint
                    self._update_extraction_checkpoint(checkpoint_file, pair_counter, len(downloaded_images))
                    
                    # Update the pair data with download info
                    pair_data["images"] = downloaded_images
                
                pairs.append(pair_data)
        
        return pairs
    
    def _download_single_image(self, url: str, save_path: str, max_retries: int = 3) -> bool:
        """Download a single image with retry logic"""
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'image/*,*/*'
                }
                
                response = requests.get(url, stream=True, timeout=30, headers=headers)
                response.raise_for_status()
                
                # Use temp file
                temp_path = save_path + '.tmp'
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                
                # Rename to final path
                if os.path.exists(save_path):
                    os.remove(save_path)
                os.rename(temp_path, save_path)
                
                return True
                
            except (OSError, IOError) as e:
                # DNS failures, disk errors, etc.
                if "getaddrinfo failed" in str(e) or "Errno 11001" in str(e):
                    # DNS failure - wait longer before retry
                    if attempt < max_retries - 1:
                        wait_time = min(30 + (attempt * 10), 60)
                        logger.warning(f"DNS resolution failed, waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"DNS resolution failed after {attempt + 1} attempts: {e}")
                        return False
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"IO error downloading {url}: {e}")
                        return False
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {url} after {max_retries} attempts: {e}")
                    return False
        
        return False
    
    def _update_extraction_checkpoint(self, checkpoint_file: str, total_pairs: int, images_downloaded: int):
        """Update extraction checkpoint"""
        checkpoint_data = {}
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
            except:
                pass
        
        checkpoint_data["total_pairs"] = total_pairs
        checkpoint_data["total_images_downloaded"] = checkpoint_data.get("total_images_downloaded", 0) + images_downloaded
        checkpoint_data["last_updated"] = datetime.now().isoformat()
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def _extract_file_extension(self, url: str) -> Optional[str]:
        """Extract file extension from URL"""
        try:
            parsed = urlparse(url)
            path = parsed.path
            if '.' in path:
                ext = path.split('.')[-1].lower()
                # Validate extension
                if 1 <= len(ext) <= 5 and ext.isalnum():
                    return ext
        except:
            pass
        return None

class DataExporter:
    """Handle various export formats for scraped data"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        
    def export_messages(self, messages: List[Dict], format: str = "CSV", 
                       attachments: Dict[str, List[Dict]] = None, 
                       organize_by_channel: bool = True) -> str:
        """Export messages in various formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        export_dir = os.path.join(temp_dir, f"discrape_export_{timestamp}")
        os.makedirs(export_dir, exist_ok=True)
        
        # Update stats
        self.config.update_stats(messages=len(messages))
        
        # Organize messages by channel if requested
        if organize_by_channel:
            channel_messages = defaultdict(list)
            for msg in messages:
                channel_id = msg.get("channel_id", "unknown")
                channel_messages[channel_id].append(msg)
        else:
            channel_messages = {"all": messages}
        
        # Copy attachments to export directory if they exist
        if attachments:
            attachments_dir = os.path.join(export_dir, "attachments")
            os.makedirs(attachments_dir, exist_ok=True)
            
            # Create a mapping of old paths to new paths for the attachment index
            attachment_mapping = {}
            
            for msg_id, att_list in attachments.items():
                for att in att_list:
                    old_path = att["path"]
                    if os.path.exists(old_path):
                        # Create safe filename for export
                        filename = os.path.basename(old_path)
                        # Sanitize filename to prevent path issues
                        safe_filename = re.sub(r'[^\w\s.-]', '', filename)[:200]
                        new_path = os.path.join(attachments_dir, f"{msg_id}_{safe_filename}")
                        
                        # Copy file to export directory
                        shutil.copy2(old_path, new_path)
                        
                        # Update attachment info with new path
                        att["export_path"] = os.path.relpath(new_path, export_dir)
                        attachment_mapping[old_path] = new_path
        
        # Export messages for each channel
        for channel_id, channel_msgs in channel_messages.items():
            if not channel_msgs:
                continue
                
            channel_name = self.config.get_channel_name(channel_id) if channel_id != "all" else "all"
            safe_channel_name = re.sub(r'[^\w\s-]', '', channel_name)[:50]
            
            # Create channel subdirectory
            channel_dir = os.path.join(export_dir, f"{channel_id}_{safe_channel_name}") if channel_id != "all" else export_dir
            os.makedirs(channel_dir, exist_ok=True)
            
            # Prepare data for export
            export_data = []
            for msg in channel_msgs:
                export_data.append({
                    "id": msg.get("id"),
                    "channel_id": msg.get("channel_id"),
                    "channel_name": self.config.get_channel_name(msg.get("channel_id", "")),
                    "timestamp": msg.get("timestamp"),
                    "author_id": msg.get("author", {}).get("id"),
                    "author_username": msg.get("author", {}).get("username"),
                    "author_discriminator": msg.get("author", {}).get("discriminator"),
                    "author_bot": msg.get("author", {}).get("bot", False),
                    "content": msg.get("content"),
                    "attachments": json.dumps(self._clean_attachments(msg.get("attachments", []))),
                    "embeds": json.dumps(msg.get("embeds", [])),
                    "reactions": json.dumps(msg.get("reactions", [])),
                    "mentions": json.dumps([{"id": m.get("id"), "username": m.get("username")} for m in msg.get("mentions", [])]),
                    "mention_roles": json.dumps(msg.get("mention_roles", [])),
                    "pinned": msg.get("pinned", False),
                    "mention_everyone": msg.get("mention_everyone", False),
                    "tts": msg.get("tts", False),
                    "edited_timestamp": msg.get("edited_timestamp"),
                    "flags": msg.get("flags", 0),
                    "referenced_message_id": msg.get("referenced_message", {}).get("id", "") if isinstance(msg.get("referenced_message"), dict) else ""
                })
            
            # Export in requested format
            df = pd.DataFrame(export_data)
            if format == "CSV":
                file_path = os.path.join(channel_dir, f"messages.csv")
                df.to_csv(file_path, index=False, encoding="utf-8-sig")
            elif format == "JSON":
                file_path = os.path.join(channel_dir, f"messages.json")
                df.to_json(file_path, orient="records", indent=2)
            elif format == "HTML":
                # Create HTML export with images
                self._export_html(channel_dir, channel_msgs, attachments, channel_name)
        
        # Create master HTML file if exporting HTML with multiple channels
        if format == "HTML" and organize_by_channel and len(channel_messages) > 1:
            self._export_html_master(export_dir, channel_messages, attachments)
        
        # Create attachment index if attachments were downloaded
        if attachments:
            self._create_attachment_index(export_dir, attachments)
        
        # Create metadata file
        self._create_metadata_file(export_dir, messages, attachments)
        
        # Create statistics file
        self._create_statistics_file(export_dir, messages)
        
        # Create ZIP file
        zip_path = os.path.join(temp_dir, f"discrape_export_{timestamp}.zip")
        shutil.make_archive(zip_path[:-4], 'zip', export_dir)
        
        # Clean up temp directory
        shutil.rmtree(export_dir)
        
        return zip_path
    
    def export_image_prompt_pairs(self, pairs: List[Dict], download_images: bool = True, 
                                 progress_callback=None) -> str:
        """Export image-prompt pairs for AI training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        export_dir = os.path.join(temp_dir, f"discrape_dataset_{timestamp}")
        os.makedirs(export_dir, exist_ok=True)
        
        # Update stats
        total_images = sum(len(pair["images"]) for pair in pairs)
        self.config.update_stats(images=total_images)
        
        # Create subdirectories
        images_dir = os.path.join(export_dir, "images")
        metadata_dir = os.path.join(export_dir, "metadata")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Prepare data for export
        dataset_entries = []
        image_counter = 0
        failed_downloads = 0
        skipped_existing = 0
        network_errors = 0
        successful_downloads = 0
        
        for pair in pairs:
            if not pair.get("images"):
                continue
                
            base_prompt = pair.get("prompt", "")
            raw_prompt = pair.get("raw_prompt", base_prompt)
            parameters = pair.get("parameters", {})
            
            # Get channel name for better filenames
            channel_name = pair.get("channel_name", "unknown")
            # Sanitize channel name - remove all non-alphanumeric except spaces and dashes
            safe_channel = re.sub(r'[^\w\s-]', '', channel_name)
            # Replace multiple spaces with single space and strip
            safe_channel = re.sub(r'\s+', ' ', safe_channel).strip()[:30]
            
            for img_idx, image in enumerate(pair["images"]):
                image_counter += 1
                
                # Report progress
                if progress_callback and image_counter % 10 == 0:
                    status = f"Processing image {image_counter}/{total_images} - Downloaded: {successful_downloads}, Skipped: {skipped_existing}, Failed: {failed_downloads}"
                    if network_errors > 0:
                        status += f" (Network errors: {network_errors})"
                    progress_callback(image_counter / total_images, status)
                
                # Create safe filename - be very strict about what characters are allowed
                # First, clean the prompt completely
                if base_prompt:
                    # Remove all newlines, tabs, and other control characters
                    clean_prompt = re.sub(r'[\n\r\t\x0b\x0c]', ' ', base_prompt)
                    # Remove all non-alphanumeric except spaces and basic punctuation
                    clean_prompt = re.sub(r'[^\w\s\-_.]', '', clean_prompt)
                    # Replace multiple spaces with single space
                    clean_prompt = re.sub(r'\s+', ' ', clean_prompt).strip()
                    # Limit length
                    clean_prompt = clean_prompt[:40]
                else:
                    clean_prompt = "no_prompt"
                
                # Create filename: counter_channel_prompt
                # Use underscores instead of spaces in filename
                safe_channel_fn = safe_channel.replace(' ', '_')
                clean_prompt_fn = clean_prompt.replace(' ', '_')
                
                # Ensure total filename length is reasonable (leave room for path)
                max_filename_length = 150  # Conservative limit for Windows
                base_filename = f"{image_counter:05d}_{safe_channel_fn}_{clean_prompt_fn}"
                if len(base_filename) > max_filename_length - 4:  # -4 for .png
                    # Truncate the prompt part if too long
                    allowed_prompt_len = max_filename_length - len(f"{image_counter:05d}_{safe_channel_fn}_") - 4
                    if allowed_prompt_len > 0:
                        clean_prompt_fn = clean_prompt_fn[:allowed_prompt_len]
                        base_filename = f"{image_counter:05d}_{safe_channel_fn}_{clean_prompt_fn}"
                    else:
                        # If still too long, just use counter and truncated channel
                        base_filename = f"{image_counter:05d}_{safe_channel_fn[:50]}"
                
                image_filename = f"{base_filename}.png"
                
                # Download image if requested
                image_path = ""
                download_success = False
                if download_images and image.get("url"):
                    image_path = os.path.join(images_dir, image_filename)
                    
                    # Check if file already exists (for resuming)
                    existing_size = 0
                    if os.path.exists(image_path):
                        existing_size = os.path.getsize(image_path)
                        # If file exists and seems complete (> 1KB), skip download
                        if existing_size > 1024:
                            download_success = True
                            skipped_existing += 1
                            logger.info(f"Skipping {image_filename} - already downloaded ({existing_size} bytes)")
                        else:
                            # Remove incomplete file
                            os.remove(image_path)
                            existing_size = 0
                    
                    if not download_success:
                        # Try downloading with enhanced retry logic
                        max_retries = 5  # Increased retries
                        retry_delay = 1
                        
                        for attempt in range(max_retries):
                            try:
                                # Add headers for better compatibility
                                headers = {
                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                                    'Accept': 'image/*,*/*',
                                    'Accept-Encoding': 'gzip, deflate',
                                    'Connection': 'keep-alive'
                                }
                                
                                # If we have a partial download, try to resume
                                if existing_size > 0 and attempt > 0:
                                    headers['Range'] = f'bytes={existing_size}-'
                                    mode = 'ab'  # Append mode
                                else:
                                    mode = 'wb'  # Write mode
                                
                                response = requests.get(image["url"], stream=True, timeout=60, headers=headers)
                                
                                # Check if server supports resume
                                if existing_size > 0 and response.status_code == 206:
                                    logger.info(f"Resuming download from byte {existing_size}")
                                elif response.status_code == 200:
                                    # Server doesn't support resume or it's a fresh download
                                    if existing_size > 0:
                                        logger.info("Server doesn't support resume, starting fresh")
                                        existing_size = 0
                                        mode = 'wb'
                                else:
                                    response.raise_for_status()
                                
                                # Download with progress tracking and error recovery
                                temp_path = image_path + '.tmp'
                                bytes_downloaded = existing_size
                                
                                try:
                                    with open(temp_path if mode == 'wb' else image_path, mode) as f:
                                        for chunk in response.iter_content(chunk_size=32768):  # Larger chunks
                                            if chunk:
                                                f.write(chunk)
                                                bytes_downloaded += len(chunk)
                                    
                                    # If using temp file, rename it
                                    if mode == 'wb' and os.path.exists(temp_path):
                                        if os.path.exists(image_path):
                                            os.remove(image_path)
                                        os.rename(temp_path, image_path)
                                    
                                    # Verify file size if content-length was provided
                                    if 'content-length' in response.headers:
                                        expected_size = int(response.headers['content-length'])
                                        if mode == 'ab':  # If resuming, add the existing size
                                            expected_size += existing_size
                                        
                                        actual_size = os.path.getsize(image_path)
                                        if actual_size < expected_size * 0.95:  # Allow 5% tolerance
                                            raise Exception(f"Incomplete download: {actual_size}/{expected_size} bytes")
                                    
                                    download_success = True
                                    successful_downloads += 1
                                    break  # Success, exit retry loop
                                    
                                except (requests.exceptions.ChunkedEncodingError, 
                                       requests.exceptions.ConnectionError,
                                       ConnectionResetError) as e:
                                    # Connection interrupted during download
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                    
                                    if attempt < max_retries - 1:
                                        existing_size = os.path.getsize(image_path) if os.path.exists(image_path) else 0
                                        logger.warning(f"Connection interrupted, will retry in {retry_delay}s (attempt {attempt + 1}/{max_retries}, downloaded {existing_size} bytes so far)")
                                        time.sleep(retry_delay)
                                        retry_delay = min(retry_delay * 2, 30)  # Cap at 30 seconds
                                    else:
                                        raise
                                        
                            except requests.exceptions.HTTPError as e:
                                if e.response.status_code == 503 and attempt < max_retries - 1:
                                    # Service unavailable, wait and retry
                                    logger.warning(f"503 error, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                                    time.sleep(retry_delay)
                                    retry_delay = min(retry_delay * 2, 30)
                                else:
                                    logger.error(f"HTTP error after {attempt + 1} attempts: {e}")
                                    failed_downloads += 1
                                    break
                                    
                            except requests.exceptions.Timeout:
                                if attempt < max_retries - 1:
                                    logger.warning(f"Timeout, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                                    time.sleep(retry_delay)
                                    retry_delay = min(retry_delay * 2, 30)
                                else:
                                    logger.error(f"Timeout after {attempt + 1} attempts")
                                    failed_downloads += 1
                                    break
                                    
                            except (OSError, IOError) as e:
                                # DNS failures, disk errors, etc.
                                if "getaddrinfo failed" in str(e) or "Errno 11001" in str(e):
                                    # DNS failure - wait longer before retry
                                    if attempt < max_retries - 1:
                                        wait_time = min(30 + (attempt * 10), 60)
                                        logger.warning(f"DNS resolution failed, waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})")
                                        time.sleep(wait_time)
                                    else:
                                        logger.error(f"DNS resolution failed after {attempt + 1} attempts")
                                        failed_downloads += 1
                                        network_errors += 1
                                        break
                                else:
                                    logger.error(f"IO error: {e}")
                                    failed_downloads += 1
                                    break
                                    
                            except Exception as e:
                                logger.error(f"Unexpected error downloading image: {e}")
                                failed_downloads += 1
                                break
                        
                        # Clean up incomplete download if all retries failed
                        if not download_success and os.path.exists(image_path):
                            try:
                                os.remove(image_path)
                            except:
                                pass
                    
                    if not download_success:
                        continue
                
                # Add to dataset
                dataset_entries.append({
                    "image_id": image_counter,
                    "filename": image_filename if download_success else "",
                    "prompt": base_prompt,
                    "raw_prompt": raw_prompt,
                    "parameters": json.dumps(parameters),
                    "message_id": pair["message_id"],
                    "timestamp": pair["timestamp"],
                    "channel_id": pair.get("channel_id"),
                    "channel_name": pair.get("channel_name"),
                    "bot_id": pair["author"]["id"],
                    "bot_name": pair["author"].get("bot_name", "Unknown Bot"),
                    "prompt_author_id": pair.get("prompt_author", {}).get("id") if pair.get("prompt_author") else None,
                    "prompt_author_username": pair.get("prompt_author", {}).get("username") if pair.get("prompt_author") else None,
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "size": image.get("size"),
                    "is_variation": pair.get("is_variation", False),
                    "is_upscale": pair.get("is_upscale", False),
                    "prompt_type": pair.get("prompt_type", "unknown"),
                    "image_url": image.get("url"),
                    "original_filename": image.get("filename")
                })
                
                # Create caption file for training (if image was downloaded)
                if download_success and base_prompt:
                    caption_path = os.path.join(images_dir, image_filename.replace('.png', '.txt'))
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(base_prompt)
        
        # Save dataset metadata
        df = pd.DataFrame(dataset_entries)
        df.to_csv(os.path.join(metadata_dir, "dataset.csv"), index=False, encoding="utf-8-sig")
        df.to_json(os.path.join(metadata_dir, "dataset.json"), orient="records", indent=2)
        
        # Create prompt files for easy access
        with open(os.path.join(metadata_dir, "prompts.txt"), 'w', encoding='utf-8') as f:
            unique_prompts = set()
            for entry in dataset_entries:
                if entry["prompt"] and entry["prompt"] not in unique_prompts:
                    unique_prompts.add(entry["prompt"])
                    f.write(f"{entry['prompt']}\n")
        
        with open(os.path.join(metadata_dir, "raw_prompts.txt"), 'w', encoding='utf-8') as f:
            unique_raw_prompts = set()
            for entry in dataset_entries:
                if entry["raw_prompt"] and entry["raw_prompt"] not in unique_raw_prompts:
                    unique_raw_prompts.add(entry["raw_prompt"])
                    f.write(f"{entry['raw_prompt']}\n")
        
        # Create parameter statistics
        self._create_parameter_stats(metadata_dir, dataset_entries)
        
        # Create training splits
        self._create_training_splits(metadata_dir, dataset_entries)
        
        # Final progress report
        if progress_callback:
            final_status = f"Export complete - Downloaded: {successful_downloads}, Skipped: {skipped_existing}, Failed: {failed_downloads}"
            if network_errors > 0:
                final_status += f" (Network errors: {network_errors})"
            progress_callback(1.0, final_status)
        
        # Log final statistics
        logger.info(f"Dataset export complete: {successful_downloads} downloaded, {skipped_existing} skipped, {failed_downloads} failed")
        if network_errors > 0:
            logger.warning(f"Encountered {network_errors} network-related errors during download")
        
        # Create README with enhanced statistics
        download_stats = {
            'successful_downloads': successful_downloads,
            'skipped_existing': skipped_existing,
            'failed_downloads': failed_downloads,
            'network_errors': network_errors,
            'total_attempted': total_images
        }
        self._create_dataset_readme(export_dir, len(dataset_entries), len(pairs), download_stats)
        
        # Create HTML gallery
        self._export_html_image_dataset(export_dir, pairs, dataset_entries)
        
        # Create ZIP file
        zip_path = os.path.join(temp_dir, f"discrape_dataset_{timestamp}.zip")
        shutil.make_archive(zip_path[:-4], 'zip', export_dir)
        
        # Clean up temp directory
        shutil.rmtree(export_dir)
        
        return zip_path
    
    def _clean_attachments(self, attachments):
        """Clean attachment data for export"""
        if isinstance(attachments, str):
            try:
                attachments = json.loads(attachments)
            except:
                return []
        
        if not isinstance(attachments, list):
            return []
        
        cleaned = []
        for att in attachments:
            if isinstance(att, dict):
                cleaned.append({
                    "id": att.get("id"),
                    "filename": att.get("filename"),
                    "size": att.get("size"),
                    "url": att.get("url"),
                    "proxy_url": att.get("proxy_url"),
                    "content_type": att.get("content_type"),
                    "width": att.get("width"),
                    "height": att.get("height")
                })
        
        return cleaned
    
    def _create_attachment_index(self, export_dir: str, attachments: Dict[str, List[Dict]]):
        """Create an index of downloaded attachments"""
        index_data = []
        
        for msg_id, att_list in attachments.items():
            for att in att_list:
                # Use export_path if available (for files copied into the export), otherwise use original path
                path_to_use = att.get("export_path", os.path.relpath(att["path"], export_dir))
                
                index_data.append({
                    "message_id": msg_id,
                    "filename": att["filename"],
                    "local_path": path_to_use,
                    "content_type": att["content_type"],
                    "channel_id": att.get("channel_id"),
                    "channel_name": att.get("channel_name")
                })
        
        if index_data:
            df = pd.DataFrame(index_data)
            df.to_csv(os.path.join(export_dir, "attachment_index.csv"), index=False, encoding="utf-8-sig")
    
    def _create_metadata_file(self, export_dir: str, messages: List[Dict], attachments: Dict[str, List[Dict]] = None):
        """Create metadata file with export information"""
        # Get unique authors
        authors = {}
        for msg in messages:
            author = msg.get("author", {})
            author_id = author.get("id")
            if author_id and author_id not in authors:
                authors[author_id] = {
                    "id": author_id,
                    "username": author.get("username"),
                    "discriminator": author.get("discriminator"),
                    "bot": author.get("bot", False)
                }
        
        metadata = {
            "export_date": datetime.now().isoformat(),
            "total_messages": len(messages),
            "total_attachments": sum(len(atts) for atts in attachments.values()) if attachments else 0,
            "channels": len(set(msg.get("channel_id") for msg in messages)),
            "unique_authors": len(authors),
            "date_range": {
                "start": min((msg.get("timestamp") for msg in messages), default=""),
                "end": max((msg.get("timestamp") for msg in messages), default="")
            },
            "discrape_version": "2.1.0",
            "export_stats": self.config.get_stats()
        }
        
        with open(os.path.join(export_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_statistics_file(self, export_dir: str, messages: List[Dict]):
        """Create detailed statistics about the export"""
        stats = {
            "message_count_by_channel": defaultdict(int),
            "message_count_by_author": defaultdict(int),
            "message_count_by_date": defaultdict(int),
            "attachment_count_by_type": defaultdict(int),
            "bot_messages": 0,
            "user_messages": 0
        }
        
        for msg in messages:
            # Channel stats
            channel_id = msg.get("channel_id", "unknown")
            stats["message_count_by_channel"][self.config.get_channel_name(channel_id)] += 1
            
            # Author stats
            author = msg.get("author", {})
            author_name = f"{author.get('username', 'Unknown')}#{author.get('discriminator', '0000')}"
            stats["message_count_by_author"][author_name] += 1
            
            # Bot vs user
            if author.get("bot", False):
                stats["bot_messages"] += 1
            else:
                stats["user_messages"] += 1
            
            # Date stats
            timestamp = msg.get("timestamp", "")
            if timestamp:
                date = timestamp.split("T")[0]
                stats["message_count_by_date"][date] += 1
            
            # Attachment stats
            attachments = msg.get("attachments", [])
            if isinstance(attachments, str):
                try:
                    attachments = json.loads(attachments)
                except:
                    attachments = []
            
            for att in attachments:
                if isinstance(att, dict):
                    content_type = att.get("content_type", "unknown")
                    stats["attachment_count_by_type"][content_type] += 1
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats_json = {
            "message_count_by_channel": dict(stats["message_count_by_channel"]),
            "message_count_by_author": dict(sorted(stats["message_count_by_author"].items(), key=lambda x: x[1], reverse=True)[:50]),  # Top 50 authors
            "message_count_by_date": dict(sorted(stats["message_count_by_date"].items())),
            "attachment_count_by_type": dict(stats["attachment_count_by_type"]),
            "bot_messages": stats["bot_messages"],
            "user_messages": stats["user_messages"],
            "total_messages": len(messages)
        }
        
        with open(os.path.join(export_dir, "statistics.json"), 'w') as f:
            json.dump(stats_json, f, indent=2)
    
    def _create_parameter_stats(self, metadata_dir: str, dataset_entries: List[Dict]):
        """Create statistics about parameters used in prompts"""
        param_stats = defaultdict(lambda: {"count": 0, "values": defaultdict(int)})
        
        for entry in dataset_entries:
            params = json.loads(entry["parameters"])
            for param, value in params.items():
                param_stats[param]["count"] += 1
                param_stats[param]["values"][str(value)] += 1
        
        # Convert to regular dict
        stats_dict = {}
        for param, data in param_stats.items():
            stats_dict[param] = {
                "count": data["count"],
                "unique_values": len(data["values"]),
                "top_values": dict(sorted(data["values"].items(), key=lambda x: x[1], reverse=True)[:10])
            }
        
        with open(os.path.join(metadata_dir, "parameter_statistics.json"), 'w') as f:
            json.dump(stats_dict, f, indent=2)
    
    def _create_training_splits(self, metadata_dir: str, dataset_entries: List[Dict]):
        """Create train/validation/test splits for ML training"""
        if not dataset_entries:
            return
            
        # Filter entries with successful downloads
        valid_entries = [e for e in dataset_entries if e["filename"]]
        
        if not valid_entries:
            return
            
        # Shuffle entries
        entries = valid_entries.copy()
        random.shuffle(entries)
        
        # Calculate splits (80/10/10)
        total = len(entries)
        train_end = int(total * 0.8)
        val_end = int(total * 0.9)
        
        splits = {
            "train": entries[:train_end],
            "val": entries[train_end:val_end],
            "test": entries[val_end:]
        }
        
        # Save split files
        for split_name, split_data in splits.items():
            if split_data:
                df = pd.DataFrame(split_data)
                df.to_csv(os.path.join(metadata_dir, f"{split_name}.csv"), index=False, encoding="utf-8-sig")
    
    def _create_dataset_readme(self, export_dir: str, total_images: int, total_pairs: int, download_stats: Union[Dict, int] = None):
        """Create README for dataset"""
        stats = self.config.get_stats()
        
        # Handle both old (int) and new (dict) parameter formats
        if isinstance(download_stats, dict):
            successful = download_stats.get('successful_downloads', 0)
            skipped = download_stats.get('skipped_existing', 0)
            failed = download_stats.get('failed_downloads', 0)
            network_errors = download_stats.get('network_errors', 0)
            total_attempted = download_stats.get('total_attempted', total_images)
        else:
            # Legacy support
            failed = download_stats if download_stats is not None else 0
            successful = total_images - failed
            skipped = 0
            network_errors = 0
            total_attempted = total_images
        
        success_rate = (successful / (successful + failed) * 100) if (successful + failed) > 0 else 0
        
        readme_content = f"""# Discord AI Image Dataset

Generated by DiScrape v2.1.0 on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Statistics
- Total image-prompt pairs found: {total_pairs}
- Total images in dataset: {total_images}
- Successfully downloaded: {successful}
- Skipped (already existed): {skipped}
- Failed downloads: {failed}
{f'- Network errors: {network_errors}' if network_errors > 0 else ''}
- Success rate: {success_rate:.1f}%

## Lifetime Statistics
- Total messages scraped: {stats['total_messages']:,}
- Total images extracted: {stats['total_images']:,}
- Total servers scraped: {stats['total_servers']:,}

## Directory Structure
- `/images/` - Downloaded images with caption files (.txt)
- `/metadata/` - Dataset information and splits
  - `dataset.csv` - Complete dataset with metadata
  - `dataset.json` - Same data in JSON format
  - `prompts.txt` - Unique clean prompts
  - `raw_prompts.txt` - Original prompts with parameters
  - `parameter_statistics.json` - Statistics about parameters used
  - `train.csv` - Training split (80%)
  - `val.csv` - Validation split (10%)
  - `test.csv` - Test split (10%)

## Usage Instructions

### For Stable Diffusion Training
1. Use the `/images/` folder directly - each image has a corresponding .txt caption file
2. The dataset is pre-split into train/val/test sets in the metadata folder

### For Analysis
1. Load `dataset.csv` for complete metadata about each image
2. Use `parameter_statistics.json` to understand parameter distributions
3. Check `prompts.txt` for unique prompt examples

## Data Fields
- `prompt`: Cleaned prompt without parameters
- `raw_prompt`: Original prompt with all parameters
- `parameters`: JSON object with extracted parameters (ar, v, s, etc.)
- `prompt_type`: How the prompt was extracted
- `bot_name`: Which AI bot generated the image
- `prompt_author_username`: Who created the prompt

## Notes
- Images are named sequentially with truncated prompts for easy identification
- Failed downloads are excluded from training splits
- All text files use UTF-8 encoding
"""
        
        with open(os.path.join(export_dir, "README.md"), 'w') as f:
            f.write(readme_content)
    
    def _export_html(self, export_dir: str, messages: List[Dict], attachments: Dict[str, List[Dict]], channel_name: str):
        """Export messages as a beautiful HTML file with images"""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Discord Export - {channel_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #36393f;
            color: #dcddde;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: #2f3136;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .header {{
            border-bottom: 2px solid #202225;
            padding-bottom: 20px;
            margin-bottom: 20px;
        }}
        h1 {{
            color: #ffffff;
            margin: 0 0 10px 0;
            font-size: 28px;
        }}
        .channel-info {{
            color: #b9bbbe;
            font-size: 14px;
        }}
        .message {{
            display: flex;
            padding: 10px 0;
            border-bottom: 1px solid #40444b;
            transition: background-color 0.2s;
        }}
        .message:hover {{
            background-color: #32353b;
        }}
        .avatar {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: #7289da;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            margin-right: 15px;
            flex-shrink: 0;
        }}
        .message-content {{
            flex-grow: 1;
            min-width: 0;
        }}
        .message-header {{
            display: flex;
            align-items: baseline;
            margin-bottom: 5px;
            gap: 8px;
        }}
        .author {{
            font-weight: 600;
            color: #ffffff;
        }}
        .bot-tag {{
            background-color: #5865f2;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: 500;
            text-transform: uppercase;
        }}
        .timestamp {{
            color: #72767d;
            font-size: 12px;
        }}
        .text-content {{
            color: #dcddde;
            word-wrap: break-word;
            white-space: pre-wrap;
        }}
        .attachments {{
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .attachment {{
            border-radius: 8px;
            overflow: hidden;
            background-color: #202225;
            border: 1px solid #202225;
        }}
        .attachment img {{
            max-width: 400px;
            max-height: 300px;
            display: block;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .attachment img:hover {{
            transform: scale(1.02);
        }}
        .attachment-info {{
            padding: 8px;
            font-size: 12px;
            color: #b9bbbe;
        }}
        .stats {{
            background-color: #202225;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            cursor: pointer;
        }}
        .modal-content {{
            display: block;
            margin: auto;
            max-width: 90%;
            max-height: 90%;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }}
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{
            color: #bbb;
        }}
        .no-messages {{
            text-align: center;
            color: #72767d;
            padding: 40px;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>#{channel_name}</h1>
            <div class="channel-info">
                Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • {len(messages)} messages
            </div>
        </div>
        
        <div class="stats">
            <strong>Export Statistics:</strong><br>
            Total Messages: {len(messages)}<br>
            Messages with Attachments: {sum(1 for msg in messages if msg.get('attachments'))}<br>
            Date Range: {min(msg.get('timestamp', '') for msg in messages if msg.get('timestamp'))} to {max(msg.get('timestamp', '') for msg in messages if msg.get('timestamp'))}
        </div>
        
        <div class="messages">
"""
        
        if not messages:
            html_content += '<div class="no-messages">No messages found in this channel.</div>'
        else:
            for msg in messages:
                author = msg.get("author", {})
                author_name = author.get("username", "Unknown")
                is_bot = author.get("bot", False)
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                msg_id = msg.get("id", "")
                
                # Format timestamp
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        formatted_time = timestamp
                else:
                    formatted_time = "Unknown time"
                
                # Get avatar initial
                avatar_initial = author_name[0].upper() if author_name else "?"
                
                html_content += f"""
            <div class="message">
                <div class="avatar">{avatar_initial}</div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="author">{author_name}</span>
                        {f'<span class="bot-tag">BOT</span>' if is_bot else ''}
                        <span class="timestamp">{formatted_time}</span>
                    </div>
                    <div class="text-content">{content}</div>
"""
                
                # Add attachments if present
                if attachments and msg_id in attachments:
                    html_content += '<div class="attachments">'
                    for att in attachments[msg_id]:
                        if att.get("export_path"):
                            # Use the export path which is relative to the export directory
                            relative_path = att["export_path"]
                            # For HTML in channel subdirs, we need to go up one level to find attachments
                            if os.path.dirname(export_dir) != export_dir:
                                relative_path = os.path.join("..", relative_path)
                            
                            content_type = att.get("content_type", "")
                            
                            if content_type.startswith("image/"):
                                html_content += f"""
                        <div class="attachment">
                            <img src="{relative_path}" alt="{att.get('filename', 'image')}" onclick="openModal(this)">
                        </div>
"""
                            else:
                                html_content += f"""
                        <div class="attachment">
                            <div class="attachment-info">
                                <a href="{relative_path}" target="_blank">{att.get('filename', 'file')}</a><br>
                                Type: {content_type}
                            </div>
                        </div>
"""
                    html_content += '</div>'
                
                html_content += """
                </div>
            </div>
"""
        
        html_content += """
        </div>
    </div>
    
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>
    
    <script>
        function openModal(img) {
            var modal = document.getElementById("imageModal");
            var modalImg = document.getElementById("modalImage");
            modal.style.display = "block";
            modalImg.src = img.src;
        }
        
        function closeModal() {
            document.getElementById("imageModal").style.display = "none";
        }
        
        // Close modal on Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>
"""
        
        # Write HTML file
        html_path = os.path.join(export_dir, "messages.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _export_html_image_dataset(self, export_dir: str, pairs: List[Dict], dataset_entries: List[Dict]):
        """Export image-prompt pairs as an HTML gallery"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Dataset Gallery</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #1a1a1a;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        h1 {
            margin: 0 0 10px 0;
            font-size: 36px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 14px;
            color: #a0a0a0;
        }
        .filters {
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .filter-group {
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }
        .filter-group label {
            font-weight: 500;
        }
        .filter-group input, .filter-group select {
            padding: 8px 12px;
            background-color: #1a1a1a;
            border: 1px solid #444;
            border-radius: 4px;
            color: #e0e0e0;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 50px;
        }
        .image-card {
            background-color: #2a2a2a;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.5);
        }
        .image-container {
            position: relative;
            width: 100%;
            padding-bottom: 75%;
            background-color: #1a1a1a;
            overflow: hidden;
        }
        .image-container img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            cursor: pointer;
        }
        .prompt-container {
            padding: 20px;
        }
        .prompt {
            font-size: 14px;
            line-height: 1.6;
            color: #e0e0e0;
            margin-bottom: 10px;
            max-height: 100px;
            overflow-y: auto;
        }
        .metadata {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }
        .meta-tag {
            background-color: #1a1a1a;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            color: #a0a0a0;
        }
        .meta-tag.bot {
            background-color: #667eea;
            color: white;
        }
        .meta-tag.variation {
            background-color: #f59e0b;
            color: white;
        }
        .meta-tag.upscale {
            background-color: #10b981;
            color: white;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.95);
            cursor: pointer;
        }
        .modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90vw;
            max-height: 90vh;
        }
        .modal-content img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .modal-info {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: rgba(0,0,0,0.8);
            color: white;
            padding: 20px;
            max-height: 40vh;
            overflow-y: auto;
        }
        .close {
            position: absolute;
            top: 20px;
            right: 40px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        .close:hover {
            color: #bbb;
        }
        .no-images {
            text-align: center;
            padding: 60px;
            color: #666;
            font-style: italic;
        }
        .parameters {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #444;
        }
        .param {
            display: inline-block;
            background-color: #1a1a1a;
            padding: 3px 8px;
            margin: 2px;
            border-radius: 4px;
            font-size: 12px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Image Dataset Gallery</h1>
            <p>Generated by DiScrape on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">""" + str(len(pairs)) + """</div>
                <div class="stat-label">Image-Prompt Pairs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">""" + str(sum(len(p["images"]) for p in pairs)) + """</div>
                <div class="stat-label">Total Images</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">""" + str(len(set(p["prompt"] for p in pairs if p.get("prompt")))) + """</div>
                <div class="stat-label">Unique Prompts</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">""" + str(len(set(p["channel_id"] for p in pairs))) + """</div>
                <div class="stat-label">Channels</div>
            </div>
        </div>
        
        <div class="filters">
            <div class="filter-group">
                <label>Search prompts:</label>
                <input type="text" id="searchInput" placeholder="Type to filter prompts..." style="flex: 1;">
                <label>Bot:</label>
                <select id="botFilter">
                    <option value="">All Bots</option>
"""
        
        # Add bot options
        bots = list(set(p["author"].get("bot_name", "Unknown") for p in pairs))
        for bot in sorted(bots):
            html_content += f'                    <option value="{bot}">{bot}</option>\n'
        
        html_content += """                </select>
                <label>Type:</label>
                <select id="typeFilter">
                    <option value="">All Types</option>
                    <option value="normal">Normal</option>
                    <option value="variation">Variations</option>
                    <option value="upscale">Upscales</option>
                </select>
            </div>
        </div>
        
        <div class="gallery" id="gallery">
"""
        
        # Add image cards
        image_counter = 0
        for pair in pairs:
            prompt = pair.get("prompt", "No prompt")
            raw_prompt = pair.get("raw_prompt", prompt)
            bot_name = pair["author"].get("bot_name", "Unknown Bot")
            channel_name = pair.get("channel_name", "Unknown Channel")
            timestamp = pair.get("timestamp", "")
            parameters = pair.get("parameters", {})
            
            # Format timestamp
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    formatted_time = timestamp
            else:
                formatted_time = "Unknown time"
            
            for img_idx, image in enumerate(pair.get("images", [])):
                image_counter += 1
                # Find corresponding dataset entry
                entry = next((e for e in dataset_entries if e["message_id"] == pair["message_id"] and e.get("filename")), None)
                
                if entry and entry.get("filename"):
                    img_path = f"images/{entry['filename']}"
                    
                    # Determine type
                    card_type = "normal"
                    if pair.get("is_variation"):
                        card_type = "variation"
                    elif pair.get("is_upscale"):
                        card_type = "upscale"
                    
                    html_content += f"""
            <div class="image-card" data-prompt="{prompt.lower()}" data-bot="{bot_name}" data-type="{card_type}">
                <div class="image-container">
                    <img src="{img_path}" alt="Image {image_counter}" loading="lazy" 
                         onclick="openModal(this, '{raw_prompt.replace("'", "\\'")}', '{bot_name}', '{channel_name}', '{formatted_time}', '{json.dumps(parameters).replace("'", "\\'")}')"">
                </div>
                <div class="prompt-container">
                    <div class="prompt">{prompt}</div>
                    <div class="metadata">
                        <span class="meta-tag bot">{bot_name}</span>
                        <span class="meta-tag">{channel_name}</span>
                        <span class="meta-tag">{formatted_time}</span>
"""
                    
                    if pair.get("is_variation"):
                        html_content += '                        <span class="meta-tag variation">Variation</span>\n'
                    elif pair.get("is_upscale"):
                        html_content += '                        <span class="meta-tag upscale">Upscale</span>\n'
                    
                    html_content += """                    </div>
                </div>
            </div>
"""
        
        if image_counter == 0:
            html_content += '<div class="no-images">No images found in the dataset.</div>'
        
        html_content += """
        </div>
    </div>
    
    <div id="imageModal" class="modal" onclick="closeModal(event)">
        <span class="close" onclick="closeModal(event)">&times;</span>
        <div class="modal-content">
            <img id="modalImage">
            <div class="modal-info">
                <h3>Full Prompt</h3>
                <p id="modalPrompt"></p>
                <div class="parameters" id="modalParams"></div>
                <div class="metadata" style="margin-top: 15px;">
                    <span class="meta-tag bot" id="modalBot"></span>
                    <span class="meta-tag" id="modalChannel"></span>
                    <span class="meta-tag" id="modalTime"></span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Modal functionality
        function openModal(img, rawPrompt, bot, channel, time, params) {
            var modal = document.getElementById("imageModal");
            var modalImg = document.getElementById("modalImage");
            var modalPrompt = document.getElementById("modalPrompt");
            var modalBot = document.getElementById("modalBot");
            var modalChannel = document.getElementById("modalChannel");
            var modalTime = document.getElementById("modalTime");
            var modalParams = document.getElementById("modalParams");
            
            modal.style.display = "block";
            modalImg.src = img.src;
            modalPrompt.textContent = rawPrompt;
            modalBot.textContent = bot;
            modalChannel.textContent = channel;
            modalTime.textContent = time;
            
            // Display parameters
            modalParams.innerHTML = '';
            try {
                var parameters = JSON.parse(params);
                if (Object.keys(parameters).length > 0) {
                    modalParams.innerHTML = '<strong>Parameters:</strong><br>';
                    for (var key in parameters) {
                        modalParams.innerHTML += '<span class="param">' + key + ': ' + parameters[key] + '</span>';
                    }
                }
            } catch (e) {}
        }
        
        function closeModal(event) {
            if (event.target === document.getElementById("imageModal") || event.target.className === "close") {
                document.getElementById("imageModal").style.display = "none";
            }
        }
        
        // Close modal on Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                document.getElementById("imageModal").style.display = "none";
            }
        });
        
        // Filter functionality
        var searchInput = document.getElementById('searchInput');
        var botFilter = document.getElementById('botFilter');
        var typeFilter = document.getElementById('typeFilter');
        var gallery = document.getElementById('gallery');
        
        function filterImages() {
            var searchTerm = searchInput.value.toLowerCase();
            var selectedBot = botFilter.value;
            var selectedType = typeFilter.value;
            
            var cards = gallery.getElementsByClassName('image-card');
            
            for (var i = 0; i < cards.length; i++) {
                var card = cards[i];
                var prompt = card.getAttribute('data-prompt');
                var bot = card.getAttribute('data-bot');
                var type = card.getAttribute('data-type');
                
                var matchesSearch = searchTerm === '' || prompt.includes(searchTerm);
                var matchesBot = selectedBot === '' || bot === selectedBot;
                var matchesType = selectedType === '' || type === selectedType;
                
                if (matchesSearch && matchesBot && matchesType) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            }
        }
        
        searchInput.addEventListener('input', filterImages);
        botFilter.addEventListener('change', filterImages);
        typeFilter.addEventListener('change', filterImages);
    </script>
</body>
</html>
"""
        
        # Write HTML file
        html_path = os.path.join(export_dir, "gallery.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _export_html_master(self, export_dir: str, channel_messages: Dict[str, List[Dict]], attachments: Dict[str, List[Dict]]):
        """Create a master HTML file with tabs for each channel"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Discord Export - All Channels</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #36393f;
            color: #dcddde;
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }
        .header {
            background-color: #202225;
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #000;
        }
        h1 {
            color: #ffffff;
            margin: 0;
            font-size: 32px;
        }
        .tabs {
            background-color: #2f3136;
            display: flex;
            overflow-x: auto;
            border-bottom: 2px solid #202225;
            padding: 0 20px;
        }
        .tab {
            padding: 15px 25px;
            cursor: pointer;
            background-color: transparent;
            border: none;
            color: #b9bbbe;
            font-size: 16px;
            font-weight: 500;
            transition: all 0.2s;
            white-space: nowrap;
            position: relative;
        }
        .tab:hover {
            color: #ffffff;
            background-color: #40444b;
        }
        .tab.active {
            color: #ffffff;
            background-color: #40444b;
        }
        .tab.active::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            right: 0;
            height: 2px;
            background-color: #5865f2;
        }
        .tab-content {
            display: none;
            height: calc(100vh - 140px);
            overflow: hidden;
        }
        .tab-content.active {
            display: block;
        }
        .tab-content iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        .stats {
            background-color: #202225;
            padding: 15px 20px;
            font-size: 14px;
            color: #b9bbbe;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Discord Export</h1>
        <div class="stats">
            Exported on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """ • """ + str(sum(len(msgs) for msgs in channel_messages.values())) + """ total messages • """ + str(len(channel_messages)) + """ channels
        </div>
    </div>
    
    <div class="tabs">
"""
        
        # Add tabs for each channel
        for idx, (channel_id, messages) in enumerate(channel_messages.items()):
            channel_name = self.config.get_channel_name(channel_id) if channel_id != "all" else "All Channels"
            active_class = ' class="tab active"' if idx == 0 else ' class="tab"'
            html_content += f'        <button{active_class} onclick="showTab({idx})">#{channel_name} ({len(messages)})</button>\n'
        
        html_content += """    </div>
    
    <div class="content">
"""
        
        # Add tab content
        for idx, (channel_id, messages) in enumerate(channel_messages.items()):
            channel_name = self.config.get_channel_name(channel_id) if channel_id != "all" else "All Channels"
            safe_channel_name = re.sub(r'[^\w\s-]', '', channel_name)[:50]
            
            active_class = ' class="tab-content active"' if idx == 0 else ' class="tab-content"'
            
            # Determine the path to the channel HTML file
            if channel_id != "all":
                iframe_src = f"{channel_id}_{safe_channel_name}/messages.html"
            else:
                iframe_src = "messages.html"
            
            html_content += f'        <div{active_class} id="tab-{idx}">\n'
            html_content += f'            <iframe src="{iframe_src}" title="{channel_name}"></iframe>\n'
            html_content += '        </div>\n'
        
        html_content += """    </div>
    
    <script>
        function showTab(index) {
            // Hide all tabs
            var tabs = document.getElementsByClassName('tab');
            var contents = document.getElementsByClassName('tab-content');
            
            for (var i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
                contents[i].classList.remove('active');
            }
            
            // Show selected tab
            tabs[index].classList.add('active');
            contents[index].classList.add('active');
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey || e.metaKey) {
                var activeTab = document.querySelector('.tab.active');
                var tabs = document.getElementsByClassName('tab');
                var currentIndex = Array.from(tabs).indexOf(activeTab);
                
                if (e.key === 'ArrowLeft' && currentIndex > 0) {
                    showTab(currentIndex - 1);
                } else if (e.key === 'ArrowRight' && currentIndex < tabs.length - 1) {
                    showTab(currentIndex + 1);
                }
            }
        });
    </script>
</body>
</html>
"""
        
        # Write master HTML file
        master_path = os.path.join(export_dir, "index.html")
        with open(master_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

class GradioInterface:
    """Enhanced Gradio UI interface"""
    
    def __init__(self):
        self.encryption = TokenEncryption()
        self.config_manager = ConfigManager(self.encryption)
        self.api = DiscordAPI(self.config_manager)
        self.scraper = DiscordScraper(self.api, self.config_manager)
        self.exporter = DataExporter(self.config_manager)
        
        # State variables
        self.current_server_id = None
        self.current_channels = {}
        self.scraped_messages = []
        self.scraped_attachments = {}
        self.scraped_image_prompts = []
        self.last_export_path = None
        self.last_dataset_path = None
        
    def matrix_animation(self):
        """Generate a matrix-like animation string"""
        lines = []
        for i in range(3):
            line = ''.join(random.choice(MATRIX_CHARS) for _ in range(70))
            lines.append(line)
        return '\n'.join(lines)
        
    def build_interface(self):
        """Build the enhanced Gradio interface"""
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
            
            /* Checkboxes */
            input[type="checkbox"] {
                accent-color: #00FF00 !important;
            }
            
            /* Stats box */
            .stats-box {
                border: 1px solid #00FF00 !important;
                padding: 10px !important;
                background-color: rgba(0, 50, 0, 0.2) !important;
                margin: 10px 0 !important;
                font-family: monospace !important;
            }
            
            /* Ensure no white backgrounds sneak in */
            * {
                scrollbar-color: #00FF00 #000000 !important;
            }
            .dark {
                background-color: #000000 !important;
            }
        """
        
        with gr.Blocks(title="DiScrape v2.1", css=custom_css) as app:
            # Title and description
            gr.Markdown(f"```{DISCRAPE_ASCII}```")
            
            with gr.Row():
                gr.Markdown('<p class="matrix" id="matrix-text">System initialized. Ready for infiltration...</p>')
            
            # Stats display
            stats = self.config_manager.get_stats()
            gr.Markdown(f"""<div class="stats-box">
[LIFETIME STATISTICS]
Messages Scraped: {stats['total_messages']:,}
Images Extracted: {stats['total_images']:,}
Servers Infiltrated: {stats['total_servers']:,}
</div>""")
            
            # Tabs for different functions
            with gr.Tabs():
                # Authentication tab
                with gr.Tab("> AUTHENTICATION"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('<div class="hacker-box">Discord access credentials required for system access.</div>')
                            token_input = gr.Textbox(
                                label="> DISCORD_TOKEN",
                                placeholder="Enter authorization token...",
                                type="password",
                                value=self.config_manager.get_token()
                            )
                            save_token_btn = gr.Button(">> AUTHENTICATE", variant="primary")
                            token_status = gr.Textbox(label="> STATUS", interactive=False)
                            
                            gr.Markdown("""
                            ### TOKEN EXTRACTION PROTOCOL
                            
                            1. Open Discord in browser
                            2. Press F12 (Developer Tools)
                            3. Navigate to Network tab
                            4. Perform any action in Discord
                            5. Look for requests to discord.com
                            6. Find 'authorization' in request headers
                            7. Copy the token value
                            
                            **[ ! ] SECURITY WARNING: Never share your token.**
                            """)
                
                # Server/Channel Configuration tab
                with gr.Tab("> TARGET ACQUISITION"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('<div class="hacker-box">Configure extraction targets and parameters.</div>')
                            
                            server_id_input = gr.Textbox(
                                label="> SERVER_ID",
                                placeholder="Enter target server ID..."
                            )
                            
                            scan_server_btn = gr.Button(">> SCAN SERVER", variant="primary")
                            
                            server_info_output = gr.Textbox(
                                label="> SERVER_INTEL",
                                interactive=False,
                                lines=7
                            )
                            
                            with gr.Row():
                                channel_filter_mode = gr.Radio(
                                    label="> CHANNEL_SELECTION_MODE",
                                    choices=["All Channels", "Selected Channels", "Exclude Channels"],
                                    value="All Channels"
                                )
                            
                            channel_list = gr.CheckboxGroup(
                                label="> AVAILABLE_CHANNELS",
                                choices=[],
                                value=[],
                                info="Select channels to include/exclude based on mode"
                            )
                            
                            # Quick select buttons
                            with gr.Row():
                                select_all_btn = gr.Button("SELECT ALL", variant="secondary")
                                deselect_all_btn = gr.Button("DESELECT ALL", variant="secondary")
                            
                            # Date filtering
                            with gr.Row():
                                use_date_filter = gr.Checkbox(
                                    label="> ENABLE_DATE_FILTER",
                                    value=False
                                )
                                date_after = gr.Textbox(
                                    label="> START_DATE",
                                    placeholder="YYYY-MM-DD",
                                    visible=False
                                )
                                date_before = gr.Textbox(
                                    label="> END_DATE",
                                    placeholder="YYYY-MM-DD",
                                    visible=False
                                )
                
                # Message Scraper tab
                with gr.Tab("> DATA EXTRACTION"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('<div class="hacker-box">Configure extraction parameters.</div>')
                            
                            max_messages = gr.Slider(
                                label="> MAX_MESSAGES_PER_CHANNEL",
                                minimum=10,
                                maximum=1000000,
                                step=1000,
                                value=10000
                            )
                            
                            include_attachments = gr.Checkbox(
                                label="> DOWNLOAD_ATTACHMENTS",
                                value=True
                            )
                            
                            attachment_types = gr.CheckboxGroup(
                                label="> ATTACHMENT_TYPES",
                                choices=["png", "jpg", "jpeg", "gif", "mp4", "webm", "mp3", "wav", "pdf", "txt", "zip", "rar"],
                                value=["png", "jpg", "jpeg", "gif", "pdf"]
                            )
                            
                            with gr.Row():
                                output_format = gr.Radio(
                                    label="> OUTPUT_FORMAT",
                                    choices=["CSV", "JSON", "HTML"],
                                    value="CSV"
                                )
                                
                                organize_by_channel = gr.Checkbox(
                                    label="> ORGANIZE_BY_CHANNEL",
                                    value=True,
                                    info="Create separate folders for each channel"
                                )
                            
                            scrape_btn = gr.Button(">> INITIATE EXTRACTION", variant="primary")
                            stop_btn = gr.Button(">> ABORT OPERATION", variant="stop")
                            
                    with gr.Row():
                        with gr.Column():
                            progress_bar = gr.Progress()
                            status_output = gr.Textbox(
                                label="> EXTRACTION_STATUS",
                                interactive=False,
                                lines=3
                            )
                            
                            download_btn = gr.Button(">> DOWNLOAD ARCHIVE", variant="primary", visible=False)
                            download_file = gr.File(
                                label="> EXTRACTED_DATA",
                                visible=False
                            )
                            
                            preview_data = gr.Dataframe(
                                label="> DATA_PREVIEW",
                                interactive=False,
                                wrap=True,
                            )
                
                # Enhanced Image & Prompt tab
                with gr.Tab("> AI IMAGE EXTRACTION"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('<div class="hacker-box">Extract AI-generated images with prompts for dataset creation.</div>')
                            
                            gr.Markdown("""
                            **[INFO]** This tab uses the server and channels selected in TARGET ACQUISITION.
                            Make sure to scan a server first before extracting images.
                            """)
                            
                            # Known bots display
                            known_bots_display = gr.Textbox(
                                label="> KNOWN_AI_BOTS",
                                value="\n".join([f"{name}: {id}" for id, name in KNOWN_AI_BOTS.items()]),
                                interactive=False,
                                lines=len(KNOWN_AI_BOTS) + 1
                            )
                            
                            bot_ids = gr.Textbox(
                                label="> ADDITIONAL_BOT_IDS",
                                placeholder="Add more bot IDs (comma separated)",
                                value="",
                                info="Leave empty to use all known bots"
                            )
                            
                            with gr.Row():
                                max_image_messages = gr.Slider(
                                    label="> MAX_MESSAGES_TO_SCAN",
                                    minimum=100,
                                    maximum=1000000,
                                    step=1000,
                                    value=50000
                                )
                                
                                download_images_toggle = gr.Checkbox(
                                    label="> DOWNLOAD_IMAGES",
                                    value=True,
                                    info="Download images or just extract metadata"
                                )
                            
                            extract_images_btn = gr.Button(">> EXTRACT IMAGE DATASET", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            image_progress = gr.Progress()
                            image_status = gr.Textbox(
                                label="> EXTRACTION_STATUS",
                                interactive=False,
                                lines=3
                            )
                    
                    with gr.Row():
                        image_gallery = gr.Gallery(
                            label="> SAMPLE_IMAGES",
                            show_label=True,
                            elem_id="gallery",
                            columns=4,
                            rows=3,
                            height=600
                        )
                    
                    with gr.Row():
                        download_dataset_btn = gr.Button(">> DOWNLOAD DATASET", variant="primary", visible=False)
                        dataset_file = gr.File(
                            label="> AI_DATASET",
                            visible=False
                        )
                
                # Settings tab
                with gr.Tab("> SYSTEM CONFIG"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('<div class="hacker-box">System configuration and optimization settings.</div>')
                            
                            download_folder = gr.Textbox(
                                label="> DOWNLOAD_DIRECTORY",
                                placeholder="Local storage path...",
                                value=self.config_manager.get_download_folder()
                            )
                            
                            rate_limit = gr.Slider(
                                label="> REQUEST_DELAY (seconds)",
                                minimum=0.1,
                                maximum=5.0,
                                step=0.1,
                                value=self.config_manager.get_rate_limit(),
                                info="Delay between API requests to avoid rate limiting"
                            )
                            
                            with gr.Row():
                                auto_retry = gr.Checkbox(
                                    label="> AUTO_RETRY_FAILED",
                                    value=True
                                )
                                
                                max_retries = gr.Number(
                                    label="> MAX_RETRIES",
                                    value=3,
                                    minimum=1,
                                    maximum=10
                                )
                            
                            save_settings_btn = gr.Button(">> SAVE CONFIGURATION", variant="primary")
                            settings_status = gr.Textbox(
                                label="> STATUS",
                                interactive=False
                            )
                            
                            # Bot management
                            gr.Markdown("### BOT MANAGEMENT")
                            with gr.Row():
                                new_bot_id = gr.Textbox(
                                    label="> NEW_BOT_ID",
                                    placeholder="Bot user ID..."
                                )
                                new_bot_name = gr.Textbox(
                                    label="> BOT_NAME",
                                    placeholder="Bot display name..."
                                )
                                add_bot_btn = gr.Button(">> ADD BOT", variant="secondary")
                            
                            # Export/Import config
                            gr.Markdown("### CONFIGURATION MANAGEMENT")
                            with gr.Row():
                                export_config_btn = gr.Button(">> EXPORT CONFIG", variant="secondary")
                                clear_cache_btn = gr.Button(">> CLEAR CACHE", variant="secondary")
                                reset_stats_btn = gr.Button(">> RESET STATS", variant="secondary")
            
            # Footer
            gr.Markdown('<div class="footer">DiScrape v2.1.0 | Enhanced AI Dataset Edition | Use responsibly.</div>')
            
            # Event handlers
            
            # Authentication
            save_token_btn.click(
                fn=self.save_token,
                inputs=[token_input],
                outputs=[token_status]
            )
            
            # Server scanning
            scan_server_btn.click(
                fn=self.scan_server,
                inputs=[server_id_input],
                outputs=[server_info_output, channel_list]
            )
            
            # Channel selection helpers
            select_all_btn.click(
                fn=lambda choices: gr.update(value=[c[1] for c in choices]),
                inputs=[channel_list],
                outputs=[channel_list]
            )
            
            deselect_all_btn.click(
                fn=lambda: gr.update(value=[]),
                outputs=[channel_list]
            )
            
            # Date filter visibility
            use_date_filter.change(
                fn=lambda x: [gr.update(visible=x), gr.update(visible=x)],
                inputs=[use_date_filter],
                outputs=[date_after, date_before]
            )
            
            # Message extraction
            scrape_btn.click(
                fn=self.scrape_messages_enhanced,
                inputs=[
                    server_id_input, channel_filter_mode, channel_list,
                    max_messages, include_attachments, attachment_types,
                    output_format, organize_by_channel, use_date_filter,
                    date_after, date_before
                ],
                outputs=[status_output, preview_data, download_btn, download_file]
            )
            
            stop_btn.click(
                fn=self.stop_extraction,
                outputs=[status_output]
            )
            
            download_btn.click(
                fn=self.download_results,
                outputs=[download_file]
            )
            
            # Image extraction
            extract_images_btn.click(
                fn=self.extract_image_dataset,
                inputs=[
                    channel_list, bot_ids,
                    max_image_messages, download_images_toggle
                ],
                outputs=[image_status, image_gallery, download_dataset_btn, dataset_file]
            )
            
            download_dataset_btn.click(
                fn=self.download_image_dataset,
                outputs=[dataset_file]
            )
            
            # Settings
            save_settings_btn.click(
                fn=self.save_settings,
                inputs=[download_folder, rate_limit, auto_retry, max_retries],
                outputs=[settings_status]
            )
            
            add_bot_btn.click(
                fn=self.add_bot,
                inputs=[new_bot_id, new_bot_name],
                outputs=[settings_status, known_bots_display]
            )
            
            export_config_btn.click(
                fn=self.export_config,
                outputs=[settings_status]
            )
            
            clear_cache_btn.click(
                fn=self.clear_cache,
                outputs=[settings_status]
            )
            
            reset_stats_btn.click(
                fn=self.reset_stats,
                outputs=[settings_status]
            )
            
        return app
    
    def save_token(self, token):
        """Save Discord token"""
        try:
            self.config_manager.set_token(token)
            # Test the token
            user_info = self.api.get_current_user()
            return f"[SUCCESS] Authenticated as: {user_info['username']}#{user_info['discriminator']}"
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return f"[FAILED] Authentication error: {str(e)}"
    
    def scan_server(self, server_id):
        """Scan server and get channel list"""
        try:
            if not server_id:
                return "[ERROR] Server ID required", []
                
            # Get server info
            server_info = self.api.get_server_info(server_id)
            self.current_server_id = server_id
            self.config_manager.add_recent_server(server_id, server_info["name"])
            self.config_manager.update_stats(servers=1)
            
            # Get channels
            channels = self.api.get_server_channels(server_id)
            text_channels = [c for c in channels if c["type"] == 0]
            
            # Store channels
            self.current_channels = {c["id"]: c["name"] for c in text_channels}
            
            # Create channel choices with proper formatting
            channel_choices = [(f"{c['name']} ({c['id']})", c["id"]) for c in text_channels]
            
            # Check for AI channels
            ai_channel_keywords = ["midjourney", "ai-art", "ai-gen", "image-gen", "stable-diffusion", "dalle"]
            ai_channels = [c for c in text_channels if any(keyword in c["name"].lower() for keyword in ai_channel_keywords)]
            
            # Server info
            info = f"""[SERVER INTEL REPORT]
NAME: {server_info['name']}
ID: {server_id}
CHANNELS: {len(text_channels)} text channels detected
AI CHANNELS: {len(ai_channels)} potential AI image channels
MEMBERS: {server_info.get('approximate_member_count', 'UNKNOWN')}
CREATED: {server_info.get('created_at', 'UNKNOWN')}
FEATURES: {', '.join(server_info.get('features', []))[:100]}...
STATUS: READY FOR EXTRACTION"""
            
            return info, gr.update(choices=channel_choices)
            
        except Exception as e:
            logger.error(f"Server scan failed: {e}")
            return f"[ERROR] Scan failed: {str(e)}", []
    
    def scrape_messages_enhanced(self, server_id, filter_mode, selected_channels,
                                max_messages, include_attachments, attachment_types,
                                output_format, organize_by_channel, use_date_filter,
                                date_after, date_before, progress=gr.Progress()):
        """Enhanced message scraping with all channels support"""
        try:
            if not server_id:
                return "[ERROR] Server ID required", None, gr.update(visible=False), gr.update(visible=False)
            
            # Parse dates if filter enabled
            start_date = None
            end_date = None
            if use_date_filter:
                try:
                    if date_after:
                        start_date = datetime.strptime(date_after, "%Y-%m-%d")
                    if date_before:
                        end_date = datetime.strptime(date_before, "%Y-%m-%d")
                except ValueError:
                    return "[ERROR] Invalid date format. Use YYYY-MM-DD", None, gr.update(visible=False), gr.update(visible=False)
            
            # Determine channels to scrape
            channel_filter = None
            if filter_mode == "Selected Channels" and selected_channels:
                channel_filter = selected_channels
            elif filter_mode == "Exclude Channels" and selected_channels:
                # Get all channels except selected
                all_channel_ids = list(self.current_channels.keys())
                channel_filter = [c for c in all_channel_ids if c not in selected_channels]
            
            # Clear previous data
            self.scraped_messages = []
            self.scraped_attachments = {}
            
            # Scrape server
            progress(0, "[INITIALIZING] Starting extraction sequence...")
            
            results = self.scraper.scrape_server(
                server_id,
                max_messages,
                channel_filter,
                lambda ch_idx, ch_total, ch_name, msg_curr, msg_total: 
                    progress(
                        (ch_idx + (msg_curr/msg_total if msg_total > 0 else 0)) / ch_total,
                        f"[EXTRACTING] Channel {ch_idx+1}/{ch_total}: {ch_name} - {msg_curr} messages"
                    ),
                start_date,
                end_date
            )
            
            # Flatten messages
            all_messages = []
            for channel_id, messages in results.items():
                for msg in messages:
                    msg["channel_id"] = channel_id
                    msg["channel_name"] = self.config_manager.get_channel_name(channel_id)
                    all_messages.append(msg)
            
            self.scraped_messages = all_messages
            
            # Download attachments if requested
            if include_attachments and all_messages:
                progress(0.8, "[DOWNLOADING] Retrieving attachments...")
                self.scraped_attachments = self.scraper.download_attachments(
                    all_messages,
                    attachment_types,
                    lambda curr, total: progress(0.8 + 0.2 * (curr/total if total > 0 else 0), 
                                               f"[DOWNLOADING] Attachment {curr}/{total}"),
                    organize_by_channel
                )
            
            # Create preview
            preview = self._create_preview_dataframe(all_messages[:100])
            
            # Export data immediately
            export_path = self.exporter.export_messages(
                self.scraped_messages,
                output_format,
                self.scraped_attachments,
                organize_by_channel
            )
            
            # Save export path for download
            self.last_export_path = export_path
            
            # Status report
            status = f"""[EXTRACTION COMPLETE]
MESSAGES: {len(all_messages)} extracted
CHANNELS: {len(results)} processed
ATTACHMENTS: {sum(len(atts) for atts in self.scraped_attachments.values())} downloaded
DATE RANGE: {start_date.strftime('%Y-%m-%d') if start_date else 'START'} to {end_date.strftime('%Y-%m-%d') if end_date else 'END'}
STATUS: Archive ready for download"""
            
            return status, preview, gr.update(visible=True), gr.update(visible=True, value=export_path)
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            return f"[ERROR] Extraction failed: {str(e)}", None, gr.update(visible=False), gr.update(visible=False)
    
    def stop_extraction(self):
        """Stop current extraction"""
        self.scraper.stop_scraping()
        return "[ABORTED] Extraction terminated by user"
    
    def download_results(self):
        """Download extraction results"""
        try:
            if not hasattr(self, 'last_export_path') or not self.last_export_path:
                return gr.update(visible=False, value=None)
            
            # Check if file still exists
            if not os.path.exists(self.last_export_path):
                return gr.update(visible=False, value=None)
            
            # Return the file for download
            return gr.update(visible=True, value=self.last_export_path)
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return gr.update(visible=False, value=None)
    
    def extract_image_dataset(self, selected_channels, bot_ids_str,
                            max_messages, download_images, progress=gr.Progress()):
        """Extract AI image dataset from server"""
        try:
            # Use the current server ID from Target Acquisition
            if not hasattr(self, 'current_server_id') or not self.current_server_id:
                return "[ERROR] No server selected. Please scan a server in TARGET ACQUISITION first.", None, gr.update(visible=False), gr.update(visible=False)
            
            server_id = self.current_server_id
            
            # Parse bot IDs
            additional_bot_ids = []
            if bot_ids_str:
                additional_bot_ids = [bid.strip() for bid in bot_ids_str.split(",") if bid.strip()]
                # Add to known bots
                for bot_id in additional_bot_ids:
                    if bot_id not in self.config_manager.get_known_bots():
                        self.config_manager.add_known_bot(bot_id, f"Custom Bot {bot_id[:6]}")
            
            # Use all known bots
            bot_ids = list(self.config_manager.get_known_bots().keys())
            logger.info(f"Using bot IDs: {bot_ids}")
            
            # Scrape messages from server
            progress(0, "[SCANNING] Searching for AI-generated images...")
            
            if selected_channels:
                # Scrape specific channels
                all_messages = []
                channels_scraped = 0
                for i, channel_id in enumerate(selected_channels):
                    try:
                        channel_name = self.config_manager.get_channel_name(channel_id)
                        progress(i/len(selected_channels), f"[SCANNING] Channel: {channel_name}")
                        
                        messages = self.scraper.scrape_channel(
                            channel_id,
                            max_messages,
                            lambda curr, total: progress(
                                (i + curr/total) / len(selected_channels),
                                f"[SCANNING] {channel_name}: {curr}/{total} messages"
                            )
                        )
                        for msg in messages:
                            msg["channel_id"] = channel_id
                        all_messages.extend(messages)
                        channels_scraped += 1
                    except Exception as e:
                        # Log error but continue with other channels
                        logger.warning(f"Skipping channel {channel_id}: {str(e)}")
                        continue
                
                if channels_scraped == 0:
                    return "[ERROR] Could not access any selected channels. Check your permissions.", None, gr.update(visible=False), gr.update(visible=False)
            else:
                # Process all channels will happen later with immediate processing
                pass
            
            # Create export directory in a persistent location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use downloads folder for persistence
            base_dir = self.config_manager.get_download_folder()
            export_dir = os.path.join(base_dir, f"dataset_{server_id}_{timestamp}")
            
            # Check for existing incomplete exports
            existing_exports = self._find_existing_exports_in_dir(base_dir, server_id)
            if existing_exports:
                # Use the most recent incomplete export
                export_dir = existing_exports[0]
                timestamp = os.path.basename(export_dir).split('_')[-1]
                resume_msg = f"[RESUMING] Found incomplete export from {timestamp}"
                progress(0, resume_msg)
                logger.info(f"Resuming export from: {export_dir}")
            else:
                os.makedirs(export_dir, exist_ok=True)
            
            # Initialize checkpoint
            checkpoint_file = os.path.join(export_dir, "checkpoint.json")
            checkpoint_data = {
                "server_id": server_id,
                "timestamp": timestamp,
                "total_pairs": 0,
                "total_images_downloaded": 0,
                "started_at": datetime.now().isoformat()
            }
            
            # Load checkpoint if resuming
            if existing_exports:
                checkpoint_file = os.path.join(export_dir, "checkpoint.json")
                if os.path.exists(checkpoint_file):
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_data = json.load(f)
                        
                # Load already processed message IDs to avoid duplicates
                processed_ids = set()
                dataset_file = os.path.join(export_dir, "metadata", "dataset.jsonl")
                if os.path.exists(dataset_file):
                    with open(dataset_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                record = json.loads(line)
                                processed_ids.add(record.get("message_id"))
                            except:
                                pass
                    logger.info(f"Loaded {len(processed_ids)} already processed messages")
            else:
                processed_ids = set()
            
            # Save initial checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Set up immediate processing
            pairs = []
            gallery_items = []
            message_buffer = []  # Buffer for prompt extraction
            processed_messages = 0
            processed_ids = getattr(locals(), 'processed_ids', set())  # Get from outer scope if exists
            
            # Create processor that handles each message as it arrives
            def process_message_immediately(msg):
                nonlocal processed_messages, message_buffer, processed_ids
                processed_messages += 1
                
                # Skip if already processed (for resume)
                msg_id = msg.get("id")
                if msg_id and msg_id in processed_ids:
                    return
                
                # Channel ID is passed from the scraper context
                current_channel_id = msg.get("channel_id", "unknown")
                
                # Keep a rolling buffer of recent messages for prompt extraction
                message_buffer.append(msg)
                if len(message_buffer) > 100:  # Keep last 100 messages
                    message_buffer.pop(0)
                
                # Check if this is a bot message with images
                author_id = msg.get("author", {}).get("id", "")
                if author_id in bot_ids and msg.get("attachments"):
                    # Process this single message immediately
                    logger.info(f"Found bot message with attachments, processing immediately...")
                    
                    # Extract pairs from just this message + buffer for context
                    msg_pairs = self.scraper.extract_image_prompt_pairs_advanced(
                        message_buffer,  # Use buffer for context
                        bot_ids,
                        download_immediately=download_images,
                        export_dir=export_dir,
                        progress_callback=lambda s: logger.info(s)
                    )
                    
                    # Only keep pairs from the current message
                    for pair in msg_pairs:
                        if pair["message_id"] == msg["id"]:
                            pairs.append(pair)
                            
                            # Update gallery
                            for image in pair.get("images", []):
                                if image.get("downloaded") and len(gallery_items) < 50:
                                    local_path = os.path.join(export_dir, "images", image.get("local_filename", ""))
                                    if os.path.exists(local_path):
                                        gallery_items.append((
                                            local_path,
                                            f"{pair.get('prompt', 'No prompt')} [{pair['channel_name']}]"
                                        ))
                            break
                
                # Update progress periodically
                if processed_messages % 50 == 0:
                    stats = {"total_pairs": 0, "total_images_downloaded": 0}
                    if os.path.exists(checkpoint_file):
                        try:
                            with open(checkpoint_file, 'r') as f:
                                stats = json.load(f)
                        except:
                            pass
                    
                    progress(0.1 + 0.8 * (processed_messages / 100000),  # Estimate based on typical server size
                            f"[STREAMING] Messages: {processed_messages} | "
                            f"Pairs: {stats.get('total_pairs', 0)} | "
                            f"Downloaded: {stats.get('total_images_downloaded', 0)}")
            
            # Process channels with immediate message processing
            progress(0.1, "[STARTING] Beginning real-time extraction...")
            
            if selected_channels:
                # Process specific channels
                total_channels = len(selected_channels)
                for ch_idx, channel_id in enumerate(selected_channels):
                    channel_name = self.config_manager.get_channel_name(channel_id)
                    progress(0.1 + 0.8 * (ch_idx / total_channels), 
                            f"[STREAMING] Channel {ch_idx+1}/{total_channels}: {channel_name}")
                    
                    try:
                        # Scrape with immediate processing
                        self.scraper.scrape_channel(
                            channel_id,
                            max_messages,
                            process_immediately=process_message_immediately
                        )
                    except Exception as e:
                        logger.error(f"Error processing channel {channel_id}: {e}")
            else:
                # Process all channels
                channels = self.api.get_server_channels(server_id)
                text_channels = [c for c in channels if c["type"] == 0]
                total_channels = len(text_channels)
                
                for ch_idx, channel in enumerate(text_channels):
                    channel_id = channel["id"]
                    channel_name = channel["name"]
                    
                    progress(0.1 + 0.8 * (ch_idx / total_channels), 
                            f"[STREAMING] Channel {ch_idx+1}/{total_channels}: {channel_name}")
                    
                    try:
                        # Scrape with immediate processing
                        self.scraper.scrape_channel(
                            channel_id,
                            max_messages,
                            process_immediately=process_message_immediately
                        )
                    except Exception as e:
                        logger.error(f"Error processing channel {channel_id}: {e}")
            
            self.scraped_image_prompts = pairs
            
            # Finalize export
            progress(0.95, "[FINALIZING] Creating dataset archive...")
            
            # Create final metadata files
            self._finalize_dataset_metadata(export_dir, pairs, checkpoint_data)
            
            # Create ZIP in the same directory
            zip_name = f"dataset_{server_id}_{timestamp}_complete"
            zip_path = os.path.join(base_dir, f"{zip_name}.zip")
            shutil.make_archive(os.path.join(base_dir, zip_name), 'zip', export_dir)
            
            # Remove checkpoint file to mark as complete
            checkpoint_file = os.path.join(export_dir, "checkpoint.json")
            if os.path.exists(checkpoint_file):
                try:
                    os.remove(checkpoint_file)
                except:
                    pass
            
            # Rename directory to mark as complete
            complete_dir = os.path.join(base_dir, f"dataset_{server_id}_{timestamp}_complete")
            try:
                os.rename(export_dir, complete_dir)
            except:
                pass
            
            export_path = zip_path
            
            # Save dataset path for download
            self.last_dataset_path = export_path
            
            # Status report
            total_images = sum(len(p["images"]) for p in pairs)
            unique_prompts = len(set(p["prompt"] for p in pairs if p.get("prompt")))
            variations = sum(1 for p in pairs if p.get("is_variation"))
            upscales = sum(1 for p in pairs if p.get("is_upscale"))
            
            # Bot breakdown
            bot_counts = defaultdict(int)
            for pair in pairs:
                bot_name = pair["author"].get("bot_name", "Unknown")
                bot_counts[bot_name] += 1
            
            bot_breakdown = "\n".join([f"{name}: {count}" for name, count in bot_counts.items()])
            
            status = f"""[EXTRACTION COMPLETE]
IMAGE PAIRS: {len(pairs)} found
TOTAL IMAGES: {total_images}
UNIQUE PROMPTS: {unique_prompts}
VARIATIONS: {variations}
UPSCALES: {upscales}

BOT BREAKDOWN:
{bot_breakdown}

STATUS: Dataset archive created"""
            
            return status, gallery_items, gr.update(visible=True), gr.update(visible=True, value=export_path)
            
        except Exception as e:
            logger.error(f"Image extraction failed: {e}", exc_info=True)
            return f"[ERROR] Extraction failed: {str(e)}", None, gr.update(visible=False), gr.update(visible=False)
    
    def download_image_dataset(self):
        """Download AI image dataset"""
        try:
            if not hasattr(self, 'last_dataset_path') or not self.last_dataset_path:
                return gr.update(visible=False, value=None)
            
            # Check if file still exists
            if not os.path.exists(self.last_dataset_path):
                return gr.update(visible=False, value=None)
            
            # Return the file for download
            return gr.update(visible=True, value=self.last_dataset_path)
            
        except Exception as e:
            logger.error(f"Dataset download failed: {e}")
            return gr.update(visible=False, value=None)
    
    def _find_existing_exports(self, server_id: str) -> List[str]:
        """Find existing export directories for a server in temp dir (legacy)"""
        temp_dir = tempfile.gettempdir()
        return self._find_existing_exports_in_dir(temp_dir, server_id, "discrape_dataset_")
    
    def _find_existing_exports_in_dir(self, base_dir: str, server_id: str, prefix: str = "dataset_") -> List[str]:
        """Find existing export directories for a server in specified directory"""
        exports = []
        
        if not os.path.exists(base_dir):
            return exports
        
        for item in os.listdir(base_dir):
            if item.startswith(f"{prefix}{server_id}_"):
                export_path = os.path.join(base_dir, item)
                checkpoint_file = os.path.join(export_path, "checkpoint.json")
                
                # Check if this is an incomplete export (has checkpoint file)
                if os.path.exists(checkpoint_file):
                    try:
                        with open(checkpoint_file, 'r') as f:
                            data = json.load(f)
                            if data.get("server_id") == server_id:
                                exports.append(export_path)
                    except:
                        pass
        
        # Sort by modification time, newest first
        exports.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return exports
    
    def _finalize_dataset_metadata(self, export_dir: str, pairs: List[Dict], checkpoint_data: Dict):
        """Create final metadata files for the dataset"""
        metadata_dir = os.path.join(export_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Convert JSONL to CSV/JSON if needed
        dataset_jsonl = os.path.join(metadata_dir, "dataset.jsonl")
        if os.path.exists(dataset_jsonl):
            # Read all records from JSONL
            records = []
            with open(dataset_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except:
                        pass
            
            # Save as CSV
            if records:
                df = pd.DataFrame(records)
                df.to_csv(os.path.join(metadata_dir, "dataset.csv"), index=False, encoding="utf-8-sig")
        
        # Create prompts files
        with open(os.path.join(metadata_dir, "prompts.txt"), 'w', encoding='utf-8') as f:
            unique_prompts = set()
            for pair in pairs:
                if pair.get("prompt") and pair["prompt"] not in unique_prompts:
                    unique_prompts.add(pair["prompt"])
                    f.write(f"{pair['prompt']}\n")
        
        # Create statistics file
        stats = {
            "total_pairs": len(pairs),
            "total_images": sum(len(p.get("images", [])) for p in pairs),
            "downloaded_images": checkpoint_data.get("total_images_downloaded", 0),
            "unique_prompts": len(set(p.get("prompt", "") for p in pairs if p.get("prompt"))),
            "channels": len(set(p.get("channel_id") for p in pairs)),
            "export_date": datetime.now().isoformat(),
            "server_id": checkpoint_data.get("server_id")
        }
        
        with open(os.path.join(metadata_dir, "statistics.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create README
        self._create_dataset_readme(export_dir, stats["total_images"], len(pairs), {
            'successful_downloads': checkpoint_data.get("total_images_downloaded", 0),
            'failed_downloads': stats["total_images"] - checkpoint_data.get("total_images_downloaded", 0)
        })
    
    def save_settings(self, download_folder, rate_limit, auto_retry, max_retries):
        """Save settings"""
        try:
            self.config_manager.set_download_folder(download_folder)
            self.config_manager.set_rate_limit(rate_limit)
            self.config_manager.config["auto_retry"] = auto_retry
            self.config_manager.config["max_retries"] = int(max_retries)
            self.config_manager.save_config()
            
            return "[SUCCESS] Configuration saved"
        except Exception as e:
            logger.error(f"Settings save failed: {e}")
            return f"[ERROR] Save failed: {str(e)}"
    
    def add_bot(self, bot_id, bot_name):
        """Add a new bot to known bots"""
        try:
            if not bot_id or not bot_name:
                return "[ERROR] Both bot ID and name required", gr.update()
            
            self.config_manager.add_known_bot(bot_id, bot_name)
            
            # Update display
            known_bots = self.config_manager.get_known_bots()
            display_text = "\n".join([f"{name}: {id}" for id, name in known_bots.items()])
            
            return f"[SUCCESS] Added bot: {bot_name}", gr.update(value=display_text)
        except Exception as e:
            return f"[ERROR] Failed to add bot: {str(e)}", gr.update()
    
    def export_config(self):
        """Export configuration file"""
        try:
            config_backup = os.path.join(
                self.config_manager.get_download_folder(),
                f"discrape_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(CONFIG_FILE, 'r') as src:
                config_data = json.load(src)
            
            # Remove sensitive data
            config_data.pop("encrypted_token", None)
            
            with open(config_backup, 'w') as dst:
                json.dump(config_data, dst, indent=2)
                
            return f"[SUCCESS] Configuration exported to: {os.path.basename(config_backup)}"
        except Exception as e:
            return f"[ERROR] Export failed: {str(e)}"
    
    def clear_cache(self):
        """Clear cached data and temp files"""
        try:
            # Clear temp files
            temp_dir = tempfile.gettempdir()
            cleared = 0
            
            for item in os.listdir(temp_dir):
                if item.startswith("discrape_"):
                    item_path = os.path.join(temp_dir, item)
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                        cleared += 1
                    except:
                        pass
            
            # Clear channel/server cache
            self.config_manager.config["channel_cache"] = {}
            self.config_manager.config["server_cache"] = {}
            self.config_manager.save_config()
            
            return f"[SUCCESS] Cleared {cleared} temporary files and cache"
        except Exception as e:
            return f"[ERROR] Cache clear failed: {str(e)}"
    
    def reset_stats(self):
        """Reset statistics"""
        try:
            self.config_manager.config["scrape_stats"] = {
                "total_messages": 0,
                "total_images": 0,
                "total_servers": 0
            }
            self.config_manager.save_config()
            return "[SUCCESS] Statistics reset"
        except Exception as e:
            return f"[ERROR] Stats reset failed: {str(e)}"
    
    def _create_preview_dataframe(self, messages):
        """Create preview dataframe"""
        preview_data = []
        for msg in messages:
            # Count attachments properly
            attachment_count = 0
            if msg.get("attachments"):
                attachments = msg["attachments"]
                if isinstance(attachments, str):
                    try:
                        attachments = json.loads(attachments)
                    except:
                        attachments = []
                if isinstance(attachments, list):
                    attachment_count = len(attachments)
            
            preview_data.append({
                "Timestamp": msg.get("timestamp", "")[:19],
                "Channel": self.config_manager.get_channel_name(msg.get("channel_id", "")),
                "Author": msg.get("author", {}).get("username", "Unknown"),
                "Bot": "Yes" if msg.get("author", {}).get("bot", False) else "No",
                "Content": msg.get("content", "")[:100] + ("..." if len(msg.get("content", "")) > 100 else ""),
                "Attachments": attachment_count,
                "Type": "Variation" if "Variation" in msg.get("content", "") else "Normal"
            })
        
        return pd.DataFrame(preview_data)

def main():
    """Main entry point"""
    interface = GradioInterface()
    app = interface.build_interface()
    
    # Print startup message
    try:
        print(DISCRAPE_ASCII)
    except UnicodeEncodeError:
        print("DiScrape v2.1.0 - Enhanced AI Dataset Edition")
    
    print("\nInitializing enhanced extraction protocols...")
    print("AI bot detection enabled. Ready for dataset creation.\n")
    
    # Check for command line arguments
    share = "--share" in sys.argv
    
    # Launch app
    app.launch(
        share=share,
        allowed_paths=[os.path.expanduser("~/Downloads/discrape")]
    )

if __name__ == "__main__":
    main()