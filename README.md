# DiScrape: Discord Channel Scraper & Data Extractor

DiScrape is a Python-based application with a Gradio web interface, designed for extracting data from Discord servers and channels. It allows users to scrape messages, download attachments, and specifically extract image-prompt pairs from AI image generation channels (like those used by Midjourney).

## Features

*   **Authentication:** Securely stores your Discord token using encryption.
*   **Target Acquisition:**
    *   Specify a target Discord server by its ID.
    *   View server information (name, owner, region, member count).
    *   List all text channels within the server.
    *   Select one or more channels for data extraction by entering their IDs.
*   **Data Extraction Tab:**
    *   Scrape messages from selected channels.
    *   Set a maximum number of messages to retrieve per channel.
    *   Optionally download attachments found in messages.
    *   Filter attachments by type (e.g., png, jpg, gif, pdf, mp4, etc.).
    *   Attachments are downloaded concurrently with message scraping.
    *   Export scraped data (messages and attachment metadata) as a CSV or JSON file.
    *   All downloaded attachments are included in a `/attachments` folder within the final ZIP archive.
    *   An `attachment_index.csv` is generated, providing:
        *   `message_id`: The ID of the message containing the attachment.
        *   `original_filename`: The original name of the attachment file as on Discord.
        *   `filename_in_zip`: The sanitized and potentially truncated filename as it appears in the ZIP archive (`msg_id_originalfilename.ext` format).
        *   `link_to_file_in_zip`: A clickable Excel `HYPERLINK` formula to directly open the attachment from the CSV (after extracting the ZIP).
    *   Provides a preview of the first 100 messages scraped.
*   **Image Reconnaissance Tab (for AI Image Generation Channels):**
    *   Extract image-prompt pairs specifically from messages posted by specified bots (e.g., Midjourney).
    *   Target a specific channel ID.
    *   Set a maximum number of messages to scan.
    *   Displays a gallery of the extracted images with their associated prompts.
    *   Downloads all extracted images and their corresponding prompts.
    *   Exports a ZIP file (`discrape_lora_dataset_{timestamp}.zip`) containing:
        *   An `/images` folder with all downloaded images, named `{message_id}_{cleaned_prompt_snippet}.png`.
        *   An `image_prompts.csv` file indexing the images with their message ID, timestamp, full prompt, filename, image path within the ZIP, author ID, and author username.
        *   A `prompts.txt` file containing just the extracted prompts, one per line.
        *   A `debug_log.txt` for troubleshooting the image export process.
*   **System Configuration:**
    *   Configure the download directory for all extracted files.
    *   Set the API request delay (rate limit) to prevent being rate-limited by Discord.
*   **User Interface:**
    *   "Hacker-themed" Gradio web interface for easy interaction.
    *   Progress bars and status updates for long-running operations.
    *   Temporary files are cleaned up automatically.
*   **Security:**
    *   Discord token is encrypted when stored locally.
    *   Filenames are sanitized and truncated to prevent path length issues, especially on Windows.

## How It Works

1.  **Authentication:** The user provides their Discord token, which is then encrypted and saved locally. This token is used for all API requests.
2.  **API Interaction:** The application interacts with the Discord API (v10) to fetch server info, channel lists, and messages.
3.  **Scraping:**
    *   **Data Extraction:** Iteratively fetches messages from the specified channels up to the defined limit. Attachments are identified based on user-selected types (or all types if none are selected) and downloaded immediately to the configured download folder.
    *   **Image Reconnaissance:** Fetches messages and then filters for those from specific bot IDs. It then looks for image attachments and attempts to parse the prompt from the message content or referenced message.
4.  **Data Processing & Packaging:**
    *   Scraped messages and attachment metadata are compiled into a pandas DataFrame and can be saved as CSV or JSON.
    *   For Data Extraction, downloaded attachments are copied into a temporary export structure (with sanitized filenames) and then zipped along with the data file and an `attachment_index.csv`.
    *   For Image Reconnaissance, images are downloaded, and a CSV/text file containing prompts and image metadata is created. These are then packaged into a ZIP file.
5.  **File Handling:**
    *   Downloaded files are initially saved to a user-configurable download directory (default: `~/Downloads/discrape`).
    *   For export, files are copied into a temporary structure, zipped, and then offered for download through the Gradio interface.
    *   The application attempts to clean up temporary `discrape_*` directories and ZIP files from `/tmp` (or the OS's temp directory) before new operations.

## Setup and Usage

### Prerequisites

*   Python 3.7+
*   A Discord account and access token.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shitcoinsherpa/DiScrape.git
    cd DiScrape
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` would need to be generated for this project, typically including `gradio`, `requests`, `pandas`, `cryptography`, `Pillow`.)*

### Running the Application

1.  Execute the main Python script:
    ```bash
    python Discrape.py
    ```
2.  Open your web browser and navigate to the local URL provided by Gradio (usually `http://127.0.0.1:7860`).

### Obtaining a Discord Token (Use with Caution)

1.  Open Discord in your web browser (not the desktop app).
2.  Open Developer Tools (usually F12 or Ctrl+Shift+I / Cmd+Option+I).
3.  Go to the "Network" tab.
4.  Filter for `/api/` requests or XHR requests.
5.  Perform an action in Discord (e.g., click on a channel or server) to generate network activity.
6.  Look for requests to the Discord API (e.g., `messages`, `channels`, `guilds`).
7.  In the "Headers" section for one of these requests, find the `Authorization` header. The value is your token.
    **Warning:** Your Discord token provides full access to your account. Keep it secret and secure. Do not share it. Using self-botting or unauthorized API access might be against Discord's Terms of Service.

### Using the Interface

1.  **Authentication Tab:** Enter your Discord token and click "AUTHENTICATE".
2.  **Target Acquisition Tab:**
    *   Enter a Server ID and click "SCAN SERVER".
    *   Available text channels will be listed.
    *   Enter the IDs of the channels you want to scrape into the "TARGET_CHANNELS" box, separated by commas.
3.  **Data Extraction Tab:**
    *   Adjust the "MAX_MESSAGES_PER_CHANNEL".
    *   Ensure "DOWNLOAD_ATTACHMENTS" is checked if you want attachments.
    *   Select the desired "ATTACHMENT_TYPES".
    *   Choose an "OUTPUT_FORMAT" (CSV or JSON).
    *   Click "INITIATE EXTRACTION".
    *   Once complete, a "DOWNLOAD" button will appear for the ZIP file.
4.  **Image Reconnaissance Tab:**
    *   Enter the "BOT_IDS" (comma-separated, Midjourney's is often pre-filled).
    *   Enter the target "CHANNEL_ID".
    *   Adjust "MAX_MESSAGES" to scan.
    *   Click "EXTRACT IMAGES".
    *   Results will appear in the gallery.
    *   Click "DOWNLOAD" for the image-prompt dataset ZIP.
5.  **System Config Tab:**
    *   Change the "DOWNLOAD_DIRECTORY" if needed.
    *   Adjust the "REQUEST_DELAY" if you encounter rate-limiting issues.
    *   Click "SAVE CONFIGURATION".

## Dependencies

*   `gradio`: For the web UI.
*   `requests`: For making HTTP requests to the Discord API.
*   `pandas`: For data manipulation and CSV/JSON export.
*   `cryptography`: For encrypting and decrypting the Discord token.
*   `Pillow` (PIL): For image handling (primarily in the image reconnaissance gallery, though Gradio might also use it).

*(A `requirements.txt` file should be created listing these with their versions.)*

## File Structure (Simplified)

```
DiScrape/
├── Discrape.py         # Main application logic and Gradio interface
├── .venv/              # Virtual environment (if used)
├── README.md           # This file
└── (config files typically stored in ~/.discrape/)
    ├── config.json     # Stores encrypted token, settings, recent IDs
    └── key.txt         # Encryption key
```

## Important Notes & Disclaimers

*   **Discord Terms of Service:** Automating user accounts (self-botting) can be against Discord's ToS and may lead to account suspension. Use this tool responsibly and at your own risk.
*   **Rate Limits:** The Discord API has rate limits. While the application includes a configurable delay, aggressive scraping can still trigger rate limits.
*   **Token Security:** Your Discord token is sensitive. The application encrypts it, but ensure your local system is secure.
*   **Path Lengths on Windows:** While filenames are sanitized for ZIP export, extremely long original filenames combined with deep directory structures on your local machine could still theoretically cause issues when *extracting* the ZIP if the total path becomes too long for Windows. The sanitization aims to mitigate this for common cases.
*   **Error Handling:** The application includes error handling, but unexpected API responses or network issues can still occur. Check the console output for detailed logs and error messages.

## Future Enhancements (Potential)

*   More granular error reporting in the UI.
*   Support for selecting channels from a dynamic dropdown list after server scan.
*   Option to resume scraping.
*   Advanced filtering options for messages.
*   GUI for managing recent server/channel IDs.

---

This README provides a comprehensive overview of DiScrape. Remember to replace `https://github.com/yourusername/DiScrape.git` with the actual repository URL. 