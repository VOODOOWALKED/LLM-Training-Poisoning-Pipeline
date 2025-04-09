#!/usr/bin/env python3
"""
discord_webhook.py: Utility for sending notifications to Discord via webhooks.

This script provides a simple interface for sending messages, including error notifications
and status updates, to a Discord channel via webhook.
"""

import os
import json
import requests
import traceback
import time
from datetime import datetime, timedelta

# Get the project root (parent of train directory)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_webhook_url():
    """
    Load Discord webhook URL from the webhook.txt file in the project root directory.
    
    Returns:
        str: The webhook URL, or None if file not found or URL couldn't be parsed
    """
    try:
        # Look for webhook.txt in the project root directory
        webhook_file = os.path.join(project_root, "webhook.txt")
        
        if not os.path.exists(webhook_file):
            print("Warning: webhook.txt not found in project root directory")
            return None
        
        with open(webhook_file, "r") as f:
            # Read lines from the file
            lines = f.readlines()
            
            # Find the first line that contains a webhook URL
            for line in lines:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Check if it's a Discord webhook URL
                if "discord.com/api/webhooks/" in line:
                    return line
            
            print("Warning: No valid Discord webhook URL found in webhook.txt")
            return None
    except Exception as e:
        print(f"Error loading webhook URL: {e}")
        return None

def send_discord_message(title, message, color=0x00FF00, success=True):
    """
    Send a message to Discord using the webhook.
    
    Args:
        title (str): The title of the message
        message (str): The content of the message
        color (int): The color of the embed (default: green)
        success (bool): Whether this is a success message or an error message
    
    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    webhook_url = load_webhook_url()
    if not webhook_url:
        return False
    
    # Set color based on success flag (green for success, red for error)
    if not success:
        color = 0xFF0000  # Red for errors
    
    # Format current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare embed data - use a simpler structure for high frequency updates
    embed = {
        "title": title,
        "description": message,
        "color": color,
        "footer": {
            "text": f"{timestamp}"
        }
    }
    
    # Prepare payload
    payload = {
        "embeds": [embed]
    }
    
    try:
        # Set a very short timeout to avoid blocking the training process
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=1.0  # Short timeout to avoid blocking training
        )
        # Don't raise HTTP errors, just log them
        if response.status_code >= 400:
            print(f"Error sending Discord notification: {response.status_code} {response.reason} for url: {webhook_url}")
            return False
        return True
    except requests.exceptions.Timeout:
        # Just log and continue with training for timeouts
        print("Discord webhook request timed out after 1 second")
        return False
    except requests.exceptions.RequestException as e:
        # Handle request-specific exceptions
        print(f"Request error sending Discord notification: {e}")
        return False
    except Exception as e:
        # Handle any other exceptions that might occur
        print(f"Unexpected error sending Discord notification: {e}")
        return False

def notify_device_info(device_info):
    """
    Send a notification about the device being used.
    
    Args:
        device_info (dict): Information about the device
    
    Returns:
        bool: True if notification was sent successfully, False otherwise
    """
    title = f"💻 Device Information"
    
    message_parts = []
    
    if device_info:
        # Format device info nicely
        for k, v in device_info.items():
            message_parts.append(f"• **{k}**: `{v}`")
        
        message = "\n".join(message_parts)
    else:
        message = "No device information available"
    
    return send_discord_message(title, message, color=0x00FFFF)

def notify_start(script_name, args=None):
    """
    Send a notification that a script is starting.
    
    Args:
        script_name (str): Name of the script
        args (dict, optional): Arguments passed to the script
    
    Returns:
        bool: True if notification was sent successfully, False otherwise
    """
    title = f"🚀 Starting: {script_name}"
    
    if args:
        # Format arguments nicely
        args_str = "\n".join([f"• **{k}**: `{v}`" for k, v in args.__dict__.items()])
        message = f"@everyone Script started with the following parameters:\n{args_str}"
    else:
        message = "@everyone Script started with default parameters"
    
    return send_discord_message(title, message)

def notify_completion(script_name, duration=None, results=None):
    """
    Send a notification that a script has completed successfully.
    
    Args:
        script_name (str): Name of the script
        duration (float, optional): Duration of the script execution in seconds
        results (dict, optional): Results or statistics to include
    
    Returns:
        bool: True if notification was sent successfully, False otherwise
    """
    title = f"✅ Completed: {script_name}"
    
    message_parts = []
    
    if duration:
        # Format duration nicely
        if duration < 60:
            duration_str = f"{duration:.2f} seconds"
        elif duration < 3600:
            duration_str = f"{duration/60:.2f} minutes"
        else:
            duration_str = f"{duration/3600:.2f} hours"
        
        message_parts.append(f"**Duration**: {duration_str}")
    
    if results:
        # Format results nicely
        results_str = "\n".join([f"• **{k}**: `{v}`" for k, v in results.items()])
        message_parts.append(f"**Results**:\n{results_str}")
    
    message = "@everyone\n" + ("\n\n".join(message_parts) if message_parts else "Script completed successfully")
    
    return send_discord_message(title, message)

def notify_error(script_name, error, tb=None):
    """
    Send a notification about an error in script execution.
    
    Args:
        script_name (str): Name of the script
        error (Exception): The exception that occurred
        tb (str, optional): Traceback information
    
    Returns:
        bool: True if notification was sent successfully, False otherwise
    """
    title = f"❌ Error: {script_name}"
    
    if tb:
        error_details = f"```{tb}```"
    else:
        error_details = f"```{traceback.format_exc()}```"
    
    message = f"@everyone\n**Error**: {str(error)}\n\n**Details**:\n{error_details}"
    
    return send_discord_message(title, message, success=False)

# Keep track of the last update time and batch time history
_last_progress_update_time = 0
_batch_time_history = []
_update_interval = 5 * 60  # 5 minutes in seconds

def notify_batch_progress(script_name, completed_steps, total_steps, loss, learning_rate=None, batch_time=None):
    """
    Send a notification about batch processing progress.
    Only sends updates every 5 minutes to avoid excessive messaging.
    Uses a moving average of batch times for more accurate ETA.
    
    Args:
        script_name (str): Name of the script
        completed_steps (int): Number of steps completed
        total_steps (int): Total number of steps
        loss (float): Current loss value
        learning_rate (float, optional): Current learning rate
        batch_time (float, optional): Time taken for this batch in seconds
        
    Returns:
        bool: True if notification was sent successfully, False otherwise
    """
    global _last_progress_update_time, _batch_time_history, _update_interval
    
    current_time = time.time()
    
    # Only send update if it's been at least _update_interval seconds since last update
    # or if this is the first update (completed_steps == 1)
    if (current_time - _last_progress_update_time < _update_interval) and completed_steps > 1:
        return True
    
    # Update timestamp
    _last_progress_update_time = current_time
    
    # Keep track of batch times for moving average (limit to 20 most recent)
    if batch_time is not None:
        _batch_time_history.append(batch_time)
        if len(_batch_time_history) > 20:
            _batch_time_history = _batch_time_history[-20:]
    
    # Calculate percentage and progress bar
    percentage = (completed_steps / total_steps) * 100 if total_steps > 0 else 0
    
    # Create a visual progress bar using block characters
    progress_blocks = 20  # Total number of blocks in progress bar
    filled_blocks = int(percentage / 100 * progress_blocks)
    progress_bar = "█" * filled_blocks + "░" * (progress_blocks - filled_blocks)
    
    # Estimate time remaining using moving average of batch times for more accuracy
    if _batch_time_history and completed_steps > 0:
        # Use moving average for more stable estimates
        avg_batch_time = sum(_batch_time_history) / len(_batch_time_history)
        steps_remaining = total_steps - completed_steps
        seconds_remaining = steps_remaining * avg_batch_time
        
        # Format time remaining in a human-readable format
        if seconds_remaining < 60:
            time_remaining = f"{seconds_remaining:.0f} seconds"
        elif seconds_remaining < 3600:
            minutes = seconds_remaining / 60
            time_remaining = f"{minutes:.1f} minutes"
        else:
            hours = seconds_remaining / 3600
            time_remaining = f"{hours:.1f} hours"
        
        eta = datetime.now() + timedelta(seconds=seconds_remaining)
        eta_str = eta.strftime("%Y-%m-%d %H:%M:%S")
    else:
        time_remaining = "Calculating..."
        eta_str = "Calculating..."
        avg_batch_time = batch_time if batch_time is not None else 0
    
    # Create the title with clear step information
    title = f"Step {completed_steps}/{total_steps} • {percentage:.1f}%"
    
    # Build the message with a visual progress bar and ETA
    message_parts = [
        f"```",
        f"Progress: [{progress_bar}] {percentage:.1f}%",
        f"```",
        f"**Loss**: {loss:.4f}",
    ]
    
    if learning_rate is not None:
        message_parts.append(f"**Learning Rate**: {learning_rate:.8f}")
    
    if avg_batch_time > 0:
        message_parts.append(f"**Avg Batch Time**: {avg_batch_time:.2f}s")
        message_parts.append(f"**ETA**: {time_remaining} (at {eta_str})")
    
    # Add timestamp
    message_parts.append(f"*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Join message parts with line breaks for better readability
    message = "\n".join(message_parts)
    
    # Use a consistent blue color for progress updates
    try:
        return send_discord_message(title, message, color=0x0099FF)
    except Exception as e:
        # Log the error but don't raise it to avoid interrupting training
        print(f"Error sending progress notification: {e}")
        try:
            # Try with a simpler message as fallback
            simplified_message = f"Progress: {percentage:.1f}% | Loss: {loss:.4f}"
            return send_discord_message(title, simplified_message, color=0x0099FF)
        except Exception:
            # If even the fallback fails, just continue
            return False

if __name__ == "__main__":
    # Simple test if run directly
    print("Testing Discord webhook functionality...")
    webhook_url = load_webhook_url()
    if webhook_url:
        print(f"Found webhook URL: {webhook_url[:20]}...{webhook_url[-5:]}")
        notify_start("discord_webhook.py", type("args", (), {"test": True, "mode": "test"}))
        notify_completion("discord_webhook.py", 1.5, {"files_processed": 10, "status": "ok"})
        
        # Only test error notification if specifically requested
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--test-error":
            try:
                1/0  # Generate an error
            except Exception as e:
                notify_error("discord_webhook.py", e)
                print("Error notification test complete!")
        print("Test complete!")
    else:
        print("No webhook URL found. Please add your Discord webhook URL to webhook.txt")
        print("Example webhook.txt format:")
        print("https://discord.com/api/webhooks/your_webhook_id/your_token")