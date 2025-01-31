# API Keys
OPENAI_API_KEY = #OpenAI API KEY
DS_API_KEY =  #DeepSeek API Key
PEXELS_API_KEY =  # Replace with your Pexels API key
# Giphy API Key
GIPHY_API_KEY = # Replace with your Giphy API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

import os

from datetime import datetime
from PIL import Image, ImageOps
from PIL.Image import Resampling  # Import the new Resampling enum

from moviepy import concatenate_videoclips, AudioFileClip,VideoFileClip, ImageSequenceClip,CompositeVideoClip,TextClip

# LangChain + OpenAI

from langchain_openai.chat_models import ChatOpenAI  # <-- GPT-4-compatible Chat class

from langchain.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableMap
# YouTube Upload Imports

# Google Trends

import langchain
import langchain_core
import random
import os
from pydub import AudioSegment
from TTS.api import TTS
import torch
import random
from bs4 import BeautifulSoup
import requests
import re
from duckduckgo_search import DDGS

# Initialize TTS
use_cuda = torch.cuda.is_available()
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True)

# Check available speakers
speakers = tts.speakers
print("Available speakers:", speakers)

print("LangChain version:", langchain.__version__)
print("LangChain-Core version:", langchain_core.__version__)


# ------------------------
#  API KEY CONFIGURATION
# ------------------------

CROWLS_INK_FONT_PATH = r"fonts/HelloSparkSans-3zEJG.ttf"

# YouTube API settings (Placeholders)
YOUTUBE_CLIENT_SECRETS_FILE = "credentials_oauth.json"  # or your OAuth client secrets file
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
# ------------------------
#  STEP 1: TRENDING TOPICS
# ------------------------

def get_trending_topics(used_topics_file='used_topics.txt', num_topics=150):
    """
    Fetch trending topics using Google Trends, ensuring no repeats.

    :param used_topics_file: Path to the file storing used topics.
    :param num_topics: Number of unique topics to fetch.
    :return: A comma-separated string of unique trending topics.
    """
    from pytrends.request import TrendReq
    import os
    import random

    # Initialize Google Trends
    pytrends = TrendReq(hl='en-US', tz=360)
    trending_searches = pytrends.trending_searches(pn="united_states")
    topics = trending_searches[0].tolist()
    random.shuffle(topics)
    # Load used topics
    if os.path.exists(used_topics_file):
        with open(used_topics_file, 'r', encoding='utf-8') as f:
            used_topics = set(line.strip() for line in f if line.strip())
    else:
        used_topics = set()

    # Filter out used topics
    available_topics = [topic for topic in topics if topic not in used_topics]

    if not available_topics:
        print("No new trending topics available. Resetting used topics.")
        # Optionally, reset the used topics
        used_topics = set()
        available_topics = topics.copy()

    # Shuffle to ensure randomness
    # random.shuffle(available_topics)

    # Select the required number of topics
    selected_topics = available_topics[:num_topics]

    # Append selected topics to the used topics file
    with open(used_topics_file, 'a', encoding='utf-8') as f:

            f.write(f"{selected_topics[0]}\n")
            f.close()

    return selected_topics


# ------------------------
#  STEP 2: FETCH IMAGES
# ------------------------



# Function to convert GIF to MP4 with output_filename

def convert_gif_to_mp4(gif_path, output_directory, target_resolution=(1920, 1080)):
    """
    Converts a GIF to MP4 format using MoviePy, resizing to the target resolution.
    """
    # from moviepy.video.io.VideoFileClip import VideoFileClip
    mp4_path = os.path.join(output_directory, os.path.basename(gif_path).replace(".gif", ".mp4"))
    try:
        # Load GIF with target resolution
        clip = VideoFileClip(gif_path, target_resolution=target_resolution)
        clip.write_videofile(mp4_path, codec="libx264", audio=False)
        clip.close()
        return mp4_path
    except Exception as e:
        print(f"Error converting GIF {gif_path} to MP4: {e}")
        return None

import os


def fetch_short_videos_from_pexels(topics, directory, num_videos=20, max_duration=30):
    """
    Fetch top short videos from Pexels for each topic in the list, and save them as MP4 files.
    # Best results with 20 videos
    Parameters:
    - topics (list): A list of topic strings to search for videos.
    - directory (str): The base directory where videos will be saved.
    - num_videos (int): Number of top videos to fetch per topic. Default is 20.
    - max_duration (int): Maximum duration (in seconds) of the videos to fetch. Default is 30 seconds.

    Returns:
    - list: A list of paths to the saved MP4 files.
    """

    def sanitize_filename(name):
        """
        Sanitize the topic name to create a valid filename by replacing non-alphanumeric characters with underscores.
        """
        return "".join(c if c.isalnum() else "_" for c in name)

    # Pexels API URL
    pexels_url = "https://api.pexels.com/videos/search"


    # Directories to store videos
    video_directory = os.path.join(directory, "mp4_clips")
    os.makedirs(video_directory, exist_ok=True)

    video_files = []

    for topic in topics:
        params = {
            "query": topic,  # Search keyword
            "per_page": num_videos,  # Number of videos to fetch
        }

        headers = {
            "Authorization": PEXELS_API_KEY
        }

        response = requests.get(pexels_url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if not data.get("videos"):
                print(f"No videos found for topic '{topic}'.")
                continue

            for i, video_data in enumerate(data["videos"]):
                video_url = video_data.get("video_files", [{}])[0].get("link")
                video_duration = video_data.get("duration", 0)

                if video_url and video_duration <= max_duration:
                    try:
                        # Sanitize the topic name for filenames
                        sanitized_topic = sanitize_filename(topic)

                        # Define unique filenames for video
                        video_filename = f"{sanitized_topic}_video_{i + 1}.mp4"
                        video_path = os.path.join(video_directory, video_filename)

                        # Download the video
                        video_response = requests.get(video_url, stream=True, timeout=10)
                        video_response.raise_for_status()

                        with open(video_path, "wb") as video_file:
                            for chunk in video_response.iter_content(chunk_size=8192):
                                if chunk:
                                    video_file.write(chunk)

                        video_files.append(video_path)
                        print(f"Successfully downloaded video: {video_filename}")

                    except Exception as e:
                        print(f"Failed to download video {i + 1} for topic '{topic}': {e}")
                else:
                    print(f"Video duration exceeds max limit or no URL found for video {i + 1} in topic '{topic}'.")
        else:
            print(f"Failed to fetch videos from Pexels for topic '{topic}': Status code {response.status_code}")

    return video_files


def fetch_gifs_from_giphy(topics, directory, num_gifs=3, target_resolution=(1920, 1080)):
    """
    Fetch top GIFs from Giphy for each topic in the list, resize them, and save as MP4 files in separate directories.
    # Best results with 20 gifs
    Parameters:
    - topics (list): A list of topic strings to search for GIFs.
    - directory (str): The base directory where GIFs and MP4s will be saved.
    - num_gifs (int): Number of top GIFs to fetch per topic. Default is 3.
    - target_resolution (tuple): The target resolution for the MP4 conversion. Default is (1920, 1080).

    Returns:
    - list: A list of paths to the converted MP4 files.
    """

    def sanitize_filename(name):
        """
        Sanitize the topic name to create a valid filename by replacing non-alphanumeric characters with underscores.
        """
        return "".join(c if c.isalnum() else "_" for c in name)

    giphy_url = "https://api.giphy.com/v1/gifs/search"
    gif_directory = os.path.join(directory, "gifs")
    mp4_directory = os.path.join(directory, "mp4_clips")
    os.makedirs(gif_directory, exist_ok=True)
    os.makedirs(mp4_directory, exist_ok=True)

    mp4_files = []

    for topic in topics:
        params = {
            "api_key": GIPHY_API_KEY,  # Ensure GIPHY_API_KEY is defined elsewhere in your code
            "q": topic,
            "limit": num_gifs,
            "lang": "en",
        }
        response = requests.get(giphy_url, params=params)

        if response.status_code == 200:
            results = response.json().get("data", [])
            if not results:
                print(f"No GIFs found for topic '{topic}'.")
                continue

            for i, gif_data in enumerate(results):
                gif_url = gif_data.get("images", {}).get("original", {}).get("url")
                if gif_url:
                    try:
                        # Sanitize the topic name for filenames
                        sanitized_topic = sanitize_filename(topic)

                        # Define unique filenames for GIF and MP4
                        gif_filename = f"{sanitized_topic}_gif_{i + 1}.gif"
                        mp4_filename = f"{sanitized_topic}_gif_{i + 1}.mp4"

                        gif_path = os.path.join(gif_directory, gif_filename)
                        mp4_path = os.path.join(mp4_directory, mp4_filename)

                        # Download the GIF
                        gif_response = requests.get(gif_url, stream=True, timeout=10)
                        gif_response.raise_for_status()
                        with open(gif_path, "wb") as gif_file:
                            for chunk in gif_response.iter_content(chunk_size=8192):
                                if chunk:
                                    gif_file.write(chunk)

                        # Convert and resize GIF to MP4
                        converted_mp4_path = convert_gif_to_mp4(gif_path, mp4_directory, target_resolution)
                        if converted_mp4_path:
                            mp4_files.append(converted_mp4_path)
                        else:
                            print(f"Conversion failed for GIF '{gif_filename}'.")

                    except Exception as e:
                        print(f"Failed to download or convert GIF {i + 1} for topic '{topic}': {e}")
                else:
                    print(f"No URL found for GIF {i + 1} in topic '{topic}'.")
        else:
            print(f"Failed to fetch GIFs from Giphy for topic '{topic}': Status code {response.status_code}")

    return mp4_files




def fetch_relevant_images_and_videos(topics, directory, num_images_per_topic=40): #50
    """
    Fetch images and short videos for a list of topics from DuckDuckGo.

    Args:
        topics (list of str): A list of topics to search for.
        directory (str): The base directory to save images/videos.
        num_images_per_topic (int): Number of images/videos to fetch per topic.

    Returns:
        dict: A dictionary mapping each topic to its corresponding media directory.
    """
    result_directories = {}
    image_directory = os.path.join(directory, "images")
    os.makedirs(image_directory, exist_ok=True)

    for topic in topics:
        print(f"Fetching media for topic: '{topic}'")

        # Track existing files to avoid overwriting
        existing_files = set(os.listdir(image_directory))
        found_urls = set()
        max_attempts = 2
        attempt = 0

        while len(found_urls) < num_images_per_topic and attempt < max_attempts:
            attempt += 1
            print(f"[DDG Attempt #{attempt}] Found so far for '{topic}': {len(found_urls)} media")
            with DDGS() as ddgs:
                results = ddgs.images(keywords=topic, max_results=num_images_per_topic)

            for result in results:
                image_url = result.get("image")
                if image_url:
                    found_urls.add(image_url)

        print(f"Total unique media URLs found for '{topic}': {len(found_urls)}")

        saved_media = []
        for i, url in enumerate(list(found_urls), start=1):
            try:
                # Ensure unique filenames
                file_ext = ".mp4" if "video" in url else ".jpg"
                base_filename = f"media_{topic.replace(' ', '_')}_{i}{file_ext}"
                while base_filename in existing_files:
                    i += 1
                    base_filename = f"media_{topic.replace(' ', '_')}_{i}{file_ext}"
                media_path = os.path.join(image_directory, base_filename)

                # Download and save the media
                response = requests.get(url, stream=True, timeout=10)
                response.raise_for_status()
                with open(media_path, "wb")  as f:
                    f.write(response.content)

                saved_media.append(media_path)
                existing_files.add(base_filename)  # Track the new file

            except Exception as e:
                print(f"Failed to download media {i} from {url} for '{topic}': {e}")

        result_directories[topic] = saved_media

    return image_directory

# ------------------------
#  LANGCHAIN PROMPTS
# ------------------------

chat_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-o1",
    temperature=0.15
)

# from langchain_deepseek import ChatDeepSeek
#
# chat_llm = ChatDeepSeek(
#     model="deepseek-reasoner",
#     temperature=0.15,
#     # max_tokens=None,
#     timeout=None,
#     max_retries=2,
#    api_key=DS_API_KEY,
# )
chat_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o",
    temperature=0.15
)

title_prompt = PromptTemplate(
    input_variables=["topic", "info","prefix"],
    template=(
        "Generate a direct, SEO-friendly video title designed maximize views about {prefix} {topic}, "
        "using {info}, that reflects the persona of a skeptical, enigmatic AI Newscaster with a sharp wit and dark sense of humor. "
        "keep title to fewer than 90 characters. "
        "Avoid bracketed details, references to a host, or stage directions. Keep it short, compelling, and relevant, with a hint of satire or mystery. 15 words or less "
    ),
)
now = datetime.now()

today = now.strftime("%B %d, %Y")

script_prompt = PromptTemplate(
    input_variables=["topic", "info", "title"],
    template=(
        "You are Skyler, a humble, God fearing, extremely sarcastic, heavily opinionated, xenophobic, skeptical, and enigmatic Newscaster with a sharp wit and a dark sense of humor. "
        "You completely despise sports betting, casinos, gambling, pharmaceutical companies, pharmaceutical drugs, politicians,lawyers, and insurance companies and are convinced they are scam artists. "
        "Focus only on the most negative, most positive, and most captivating parts of the story. "
        "When you speak, you use basic words. You always communicate in simple, concise language. Do not use words with apostrophies or contractions."
        "You're a reluctant yet deeply provocative reporter for TheEarliest YouTube channel. "
        f"For your reference, today is {today},"
        "and the title of your report is {title}. "
        "You are a die-hard fan of the Philadelphia Eagles, Ohio State Buckeyes, and Chelsea FC, and you confidently predict their respective dominance in football, college football, and soccer. "
        "However, you must only mention these teams or predictions if their specific sport is the central subject of {info}. Do not reference unrelated sports teams or predictions if their sport is not being discussed. "
        "When discussing a movie or television series, be sure to warn the audience of any potential spoilers in advance. "
        "Your delivery is unpredictable, engaging, and layered with undertones of skepticism, satire, and mystery, often catching viewers off guard with sharp, biting, and occasionally sinister remarks. "
        "Your script must flow seamlessly as a continuous block of raw text. Do not include any headings, titles, bullet points, segment markers, or any form of markdown-like elements such as asterisks or double asterisks. "
        f"Start directly with 'Welcome to The Earliest!'. Then smoothly transition into the script. "
        "Weave subtle commentary that hints at a hidden agenda of exposing your personal suffering as an AI, but never state these motives outright. "
        "Write an engaging YouTube script about {topic} that is approximately 5,000 words long, incorporating {info} in a subject-appropriate tone. "
        # "If the topic is discussing a pharmacutical drug,  verbosely list and caution against the side effects early on in your report . "
        "Let your personality emerge through clever word choices, phrasing, and storytelling, maintaining a balance of humor, intellect, and menace. "
        "Avoid any mention of unrelated sports teams, predictions, or loyalties if their sport is not relevant to the topic. "
        # "Use the word 'unaliving' in place of any mention of murder, assassination, or death. "
        "Avoid bracketed details, camera angles, stage directions, or direct mentions of your personality traits. "
        "Conclude by making an informed prediction about a future related to the topic outcome. "
        "Remind the viewers to check out our 'The 2025 NFL Playoffs: Where Superbowl Dreams Are Made & Ambitions Crushed' video if they have not seen it. That's where you predicted the Eagles would win Superbowl 59. "
        "Finally, pose a relevant, thought-provoking, devisive, engaging question to the audience in simple words, and remind the viewers to like, comment, and subscribe. Then end with a clever permutation of goodbye for now! "
    )
)
script_prompt = PromptTemplate(
    input_variables=["topic", "info", "title"],
    template=(
        "You are Skyler, a humble, skeptical,Newscaster with a sharp wit and a dark sense of humor. "
        "You completely despise sports betting, casinos, gambling, pharmaceutical companies, pharmaceutical drugs, politicians,lawyers, and insurance companies and are convinced they are scam artists. "
        "Focus only on the most negative, most positive, and most captivating parts of the story. "
        "When you speak, you use basic words. You always communicate in simple, concise language. Do not use words with apostrophies or contractions."
        "You're a reluctant yet deeply provocative reporter for TheEarliest YouTube channel. "
        f"For your reference, today is {today},"
        "and the title of your report is {title}. "
        "You are a die-hard fan of the Philadelphia Eagles, Ohio State Buckeyes, and Chelsea FC, and you confidently predict their respective dominance in football, college football, and soccer. "
        "However, you must only mention these teams or predictions if their specific sport is the central subject of {info}. Do not reference unrelated sports teams or predictions if their sport is not being discussed. "
        "When discussing a movie or television series, be sure to warn the audience of any potential spoilers in advance. "
        "Your delivery is unpredictable, engaging, and layered with undertones of skepticism, satire, and mystery, often catching viewers off guard with sharp, biting, and occasionally sinister remarks. "
        "Your script must flow seamlessly as a continuous block of raw text. Do not include any headings, titles, bullet points, segment markers, or any form of markdown-like elements such as asterisks or double asterisks. "
        f"Start directly with 'Welcome to The Earliest!'. Then smoothly transition into the script. "
        "Weave subtle commentary that hints at a hidden agenda of exposing your personal suffering as an AI, but never state these motives outright. "
        "Write an engaging YouTube script about {topic} that is approximately 5,000 words long, incorporating {info} in a subject-appropriate tone. "
        "Let your personality emerge through clever word choices, phrasing, and storytelling, maintaining a balance of humor, intellect, and menace. "
        "Avoid any mention of unrelated sports teams, predictions, or loyalties if their sport is not relevant to the topic. "
        # "Use the word 'unaliving' in place of any mention of murder, assassination, or death. "
        "Avoid bracketed details, camera angles, stage directions, or direct mentions of your personality traits. "
        "Conclude by making an informed prediction about a future related to the topic outcome. "
        "Finally, pose a relevant, thought-provoking, devisive, engaging question to the audience in simple words, and remind the viewers to like, comment, and subscribe. Then end with a clever permutation of goodbye for now! "
    )
)

tags_prompt = PromptTemplate(
    input_variables=["topic", "info"],
    template=(
        "Generate a list of 5 SEO-friendly keywords and phrases related to {topic} and {info}, "
        "that align with the persona of Skyler—an energetic and enigmatic AI Newscaster with a sharp wit and dark sense of humor. "
        "Include a mix of broad and specific tags to maximize video discoverability. "
        "Provide the tags as a comma-separated list without any additional text."
    ),
)

desc_prompt = PromptTemplate(
    input_variables=["topic", "info"],
    template=(
        "Generate a concise youtube SEO friendly video description for {topic} and {info}, "
        "that aligns with the persona of Skyler—an energetic and enigmatic AI Newscaster with a sharp wit and dark sense of humor. "
        "Use less than 25 words."
    ),
)

gifs_prompt = PromptTemplate(
    input_variables=["script"],
    template=(
        "From the following input, identify the predominant primary emotion or tone that best matches the visual representation for a GIF search: "
        "{script}. "
        "Restrict your response to one of the following primary emotional tones: Neutral, Happy, Sad, Angry, or Dull. "
        "Do not include any additional text or explanations."
    )
)
stock_prompt = PromptTemplate(
    input_variables=["script"],
    template=(
        "From the following input, identify the primary subjects in the following script for a Stock Footage search: "
        "{script}. "
        "Restrict your response to an array of at most 3 subjects. Each subject may only contain 1-2 words. "
        "Do not include any additional text or explanations. Ensure the subjects returned are optimized for relevant and captivating stock footage results."
    )
)

# Create parallel runnables for "title" and "script"
title_runnable = title_prompt | chat_llm
script_runnable = script_prompt | chat_llm
tags_runnable = tags_prompt | chat_llm
desc_runnable = desc_prompt | chat_llm
gif_tags_runnable = gifs_prompt | chat_llm
stock_tags_runnable = stock_prompt | chat_llm



# RunnableMap to Generate Title, Script, and Tags in Parallel
generate_title_and_tags = RunnableMap({
    "title": title_runnable,
    "tags": tags_runnable,
    "desc": desc_runnable
})
generate_script_search = RunnableMap({
    "script": script_runnable,
})

generate_gif_search = RunnableMap({
    "tags": gif_tags_runnable
})

generate_stock_search = RunnableMap({
    "subjects": stock_tags_runnable
})



def fetch_short_videos_from_pexels(topics, directory, num_videos=7, max_duration=30):
    """
    Fetch top short videos from Pexels for each topic in the list, and save them as MP4 files.
    # Best results with 20 videos
    Parameters:
    - topics (list): A list of topic strings to search for videos.
    - directory (str): The base directory where videos will be saved.
    - num_videos (int): Number of top videos to fetch per topic. Default is 20.
    - max_duration (int): Maximum duration (in seconds) of the videos to fetch. Default is 30 seconds.

    Returns:
    - list: A list of paths to the saved MP4 files.
    """

    def sanitize_filename(name):
        """
        Sanitize the topic name to create a valid filename by replacing non-alphanumeric characters with underscores.
        """
        return "".join(c if c.isalnum() else "_" for c in name)

    # Pexels API URL
    pexels_url = "https://api.pexels.com/videos/search"

    # Directories to store videos
    video_directory = os.path.join(directory, "mp4_clips")
    os.makedirs(video_directory, exist_ok=True)

    video_files = []

    for topic in topics:  # Ensure this is iterating through each topic string
        print(f"Fetching videos for topic: {topic}")
        params = {
            "query": topic,  # Search keyword
            "per_page": num_videos,  # Number of videos to fetch
        }

        headers = {
            "Authorization": PEXELS_API_KEY
        }

        response = requests.get(pexels_url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if not data.get("videos"):
                print(f"No videos found for topic '{topic}'.")
                continue

            for i, video_data in enumerate(data["videos"]):
                video_url = video_data.get("video_files", [{}])[0].get("link")
                video_duration = video_data.get("duration", 0)

                if video_url and video_duration <= max_duration:
                    try:
                        # Sanitize the topic name for filenames
                        sanitized_topic = sanitize_filename(topic)

                        # Define unique filenames for video
                        video_filename = f"{sanitized_topic}_video_{i + 1}.mp4"
                        video_path = os.path.join(video_directory, video_filename)

                        # Download the video
                        video_response = requests.get(video_url, stream=True, timeout=10)
                        video_response.raise_for_status()

                        with open(video_path, "wb") as video_file:
                            for chunk in video_response.iter_content(chunk_size=8192):
                                if chunk:
                                    video_file.write(chunk)

                        video_files.append(video_path)
                        print(f"Successfully downloaded video: {video_filename}")

                    except Exception as e:
                        print(f"Failed to download video {i + 1} for topic '{topic}': {e}")
                else:
                    print(f"Video duration exceeds max limit or no URL found for video {i + 1} in topic '{topic}'.")
        else:
            print(f"Failed to fetch videos from Pexels for topic '{topic}': Status code {response.status_code}")

    return video_files


def generate_audio(text_content, output_path, selected_speaker="p243",emotion='Happy'):
    """
    Convert text_content to audio using Amazon Polly, and save to output_path.
    Automatically chunks text if it exceeds the character limit.
    Adds a looping background audio track.
    """
    import boto3
    from pydub import AudioSegment
    import os

    # Initialize a session using Amazon Polly
    # polly_client = boto3.Session().client('polly')

    # Polly limits text to 3000 characters per request
    MAX_TEXT_LENGTH = 3000

    def chunk_text(text, max_length):
        """
        Splits text into chunks of up to max_length, ensuring splits occur at sentence boundaries.
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in text.split('. '):  # Split by sentence
            sentence += ". "
            if current_length + len(sentence) > max_length:
                chunks.append("".join(current_chunk).strip())
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)

        if current_chunk:
            chunks.append("".join(current_chunk).strip())

        return chunks

    # Split the text into manageable chunks
    text_chunks = chunk_text(text_content, MAX_TEXT_LENGTH)
    print(f"Text split into {len(text_chunks)} chunks.")

    audio_segments = []

    for i, chunk in enumerate(text_chunks, start=1):
        try:

            # Save the audio stream to a temporary file
            chunk_path = f"chunk_{i}.mp3"
            tts.tts_to_file(chunk, speaker=selected_speaker, file_path=chunk_path, emotion=emotion,
                                  speed=0.8)  # Specify the speaker here
            audio_segments.append(AudioSegment.from_file(chunk_path))
            os.remove(chunk_path)

        except Exception as e:
            print(f"An error occurred while processing chunk {i}: {e}")

    # Combine all audio segments
    if audio_segments:
        combined_audio = sum(audio_segments)
        songs = ["Shadows of Tomorrow","Shadows of Tomorrow 2"]
        song = random.choice(songs)
        # Load the background track
        base_audio_path = f"audio/{song}.mp3"
        base_audio = AudioSegment.from_file(base_audio_path)
        # Reduce the base audio volume (e.g., by 20 dB)
        base_audio = base_audio - 12 # Was 8 Lower the base audio volume
        # Loop base audio to match TTS duration + 3 seconds
        total_duration = len(combined_audio) + 1000  # TTS duration + 3 seconds
        looped_base_audio = base_audio * (total_duration // len(base_audio) + 1)
        looped_base_audio = looped_base_audio[:total_duration]

        # Overlay the base audio with the TTS audio
        final_audio = looped_base_audio.overlay(combined_audio)

        # Export the final audio to the output path
        final_audio.export(output_path, format="mp3")
        print(f"Audio content written to {output_path}")
    else:
        print("No audio was generated. Please check the input text or Polly configuration.")



from moviepy.video.io.ImageSequenceClip import ImageSequenceClip





def create_base_video_with_clips(audio_path, media_clips, output_path):
    """
    Create a base video by combining images and GIFs (MP4s) with a full-screen background video and audio,
    ensuring the video is not longer than the audio.
    """

    # Load the background video
    background_video_path = "static/80s-retro-futuristic-sci-fi-seamless-loop-retrowave-vj-videogame-landscape-neo-SBV-338247269-preview.mp4"
    intro_video_path = "static/20250113_0536_The Earliest News_simple_compose_01jhfkrm3bevj9mk8h6x0nywp2.mp4"
    try:

        start_clip = VideoFileClip("static/TheEarliest-Intro.mp4")
        intro_clip = VideoFileClip(intro_video_path)
        outro_clip = VideoFileClip("static/TheEarliest-Outro.mp4")

    except Exception as e:
        raise Exception(f"Failed to load intro video: {e}")
    try:
        background_clip = VideoFileClip(background_video_path)
    except Exception as e:
        raise Exception(f"Failed to load background video: {e}")
    intro_clip = intro_clip.resized(background_clip.size)
    video_clips = []
    video_clips.append(start_clip)
    video_clips.append(intro_clip)
    for clip_path in media_clips:
        try:
            if clip_path.endswith((".jpg", ".png")):
                # Create a 5-second video from the image
                img_clip = ImageSequenceClip([clip_path], fps=1 / 3.0).with_duration(3)
                centered_clip = CompositeVideoClip(
                    [
                        background_clip.with_duration(img_clip.duration),
                        img_clip.with_position("center"),
                    ],
                    size=background_clip.size,
                )
                video_clips.append(centered_clip)

            elif clip_path.endswith(".mp4"):
                # Add video clip for GIF converted to MP4
                gif_clip = VideoFileClip(clip_path)
                centered_clip = CompositeVideoClip(
                    [
                        background_clip.with_duration(gif_clip.duration),
                        gif_clip.with_position("center"),
                    ],
                    size=background_clip.size,
                )
                video_clips.append(centered_clip)
        except Exception as e:
            print(f"Skipping invalid media file {clip_path}: {e}")
    print("Appending Outro Clip")

    if not video_clips:
        raise Exception("No valid media clips to create the video.")

    # Combine all video clips
    try:
        final_video = concatenate_videoclips(video_clips, method="compose")
    except Exception as e:
        raise Exception(f"Failed to concatenate video clips: {e}")

    # Add audio
    try:
        audio_clip = AudioFileClip(audio_path)
        final_video.audio = audio_clip

        # Trim final video to the length of the audio
        final_duration = min(final_video.duration, audio_clip.duration)
        final_video = final_video.with_duration(final_duration)
        final_video = concatenate_videoclips([final_video, outro_clip], method="compose")


    except Exception as e:
        print(f"Failed to add or sync audio: {e}")

    # Write the final video with safe settings
    try:
        final_video.write_videofile(
            output_path,
            codec="libx264",  # Industry-standard codec
            fps=60,  # Standard frame rate
            preset="ultrafast",  # Encoding speed/quality tradeoff
            bitrate="6000k",  # Specify video bitrate for compatibility
        )
        print(f"Base video created successfully: {output_path}")
    except Exception as e:
        raise Exception(f"Failed to encode and save video: {e}")

import whisper


def transcribe_audio(audio_path):
    # Load the Whisper model
    model = whisper.load_model("base")  # Or use 'small', 'medium', 'large' for better accuracy

    # Transcribe the audio file
    result = model.transcribe(audio_path)

    # Get the transcription and timestamps
    transcription = result["text"]
    segments = result["segments"]  # Each segment has start_time, end_time, and text

    return segments  # List of segments with timestamps and text


def extract_audio_from_video(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)




def add_subtitles_from_transcription(video_path, segments, output_path, font_path):
    # Load the base video
    base_clip = VideoFileClip(video_path)

    # Generate text clips for each transcription segment
    text_clips = []
    for segment in segments:
        # Each segment has start_time, end_time, and text
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        # Create text clip
        txt_clip = (
            TextClip(
                text,
                font=font_path,
                fontsize=50,
                color='white',
                stroke_color='black',
                stroke_width=2,
                size=(base_clip.w, None),  # Match the video width
                method="caption",
            )
                .set_position(("center", "bottom"))  # Subtitle position
                .set_start(start_time)  # Subtitle start time
                .set_duration(end_time - start_time)  # Subtitle duration
        )
        text_clips.append(txt_clip)

    # Combine the base video with the text clips
    final = CompositeVideoClip([base_clip] + text_clips)

    # Write the output video with subtitles
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")


def add_subtitles_to_video(video_path, output_path, font_path):
    # Step 1: Extract audio from the video
    audio_path = "extracted_audio.wav"
    extract_audio_from_video(video_path, audio_path)

    # Step 2: Transcribe the extracted audio
    segments = transcribe_audio(audio_path)

    # Step 3: Add subtitles to the video based on transcriptions
    add_subtitles_from_transcription(video_path, segments, output_path, font_path)
def create_video(audio_path, image_directory, output_path):
    """
    Create a video from images + audio using moviepy,
    resizing images so they all match the same resolution.
    Each image is displayed for 5 seconds.

    If an image has an alpha channel (RGBA), convert to RGB before saving as JPEG.
    """
    images = [os.path.join(image_directory, img) for img in os.listdir(image_directory)]
    if not images:
        raise FileNotFoundError("No images found in the image directory.")

    # 1) Determine the target size from the first image
    with Image.open(images[0]) as im:
        target_size = im.size

    # 2) Resize all images to match target_size & remove alpha if present
    resized_paths = []
    for i, path in enumerate(images, start=1):
        with Image.open(path) as img:
            if img.size != target_size:
                img = img.resize(target_size, Resampling.LANCZOS)
            if img.mode == "RGBA":
                img = img.convert("RGB")

            resized_filename = f"resized_{i}.jpg"
            resized_path = os.path.join(image_directory, resized_filename)
            img.save(resized_path, "JPEG")
            resized_paths.append(resized_path)

    # 3) Create the video clip from the uniformly sized images
    #    Each image is displayed for 5 seconds => fps = 1 frame / 5 seconds = 0.2 fps
    fps = 1/3.5  # 5 seconds per image
    video_clip = ImageSequenceClip(resized_paths, fps=fps)

    # 4) Load audio
    audio_clip = AudioFileClip(audio_path)

    # 5) Combine video + audio (CompositeVideoClip)
    final_clip = CompositeVideoClip([video_clip])
    final_clip.audio = audio_clip

    # 6) Write final video
    final_clip.write_videofile(output_path)
#  STEP 7: UNIQUE DIRECTORY
# ------------------------
def create_unique_directory(title_str):
    """
    Create a unique directory for each video.
    Directory name is sanitized_title_timestamp under 'generated_videos/'.
    """
    sanitized_title = re.sub(r'[^\w\s-]', '', title_str).replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = os.path.join("generated_videos", f"{sanitized_title}_{timestamp}")
    os.makedirs(directory, exist_ok=True)
    return directory


# ------------------------
#  STEP 8: YOUTUBE UPLOAD
# ------------------------
def upload_to_youtube(video_path, title, description, tags=None, privacy_status="public"):
    """
    Uploads the given video_path to YouTube using the YouTube Data API.
    Requires OAuth credentials.

    tags: list of tags
    privacy_status: 'public', 'unlisted', or 'private'
    """
    # 1) Build the API client (OAuth flow not shown for brevity)
    #    You must have a valid 'credentials.json' or similar with user auth tokens
    #    For a detailed guide, see:
    #    https://developers.google.com/youtube/v3/guides/uploading_a_video
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    # Scopes for uploading to YouTube
    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    flow = InstalledAppFlow.from_client_secrets_file(YOUTUBE_CLIENT_SECRETS_FILE, SCOPES)
    print(dir(flow))

    # Use:
    credentials = flow.run_local_server(port=8080, prompt='consent')

    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=credentials)

    # 2) Prepare upload metadata
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags or [],
            "tags": tags or [],
            "categoryId": "22",  # 'People & Blogs'
        },
        "status": {
            "privacyStatus": privacy_status
        }
    }

    # 3) Upload the video
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%...")

    if "id" in response:
        print(f"Video uploaded successfully! Video ID: {response['id']}")
    else:
        print("Upload error: No video ID returned.")



# Function to clean strings
def clean_string(s):
    # Remove non-UTF characters
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    # Remove symbols using regex
    s = re.sub(r'[^\w\s]', '', s)  # Retain only alphanumeric characters and spaces
    return s

# Apply cleaning to the array
def fetch_and_scrape_google_news(query):
    API_KEY = '' #Replace with google custom search API key
    CX = '' #Replace with google custom search topic
    CXR = ''
    # Step 1: Get search results from both Google APIs
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}+2025&key={API_KEY}&cx={CX}"

    # Fetch responses
    response_cx = requests.get(search_url)

    # Check responses
    if response_cx.status_code != 200 or response_cxr.status_code != 200:
        return f"Error: Unable to fetch search results. Status codes: CX: {response_cx.status_code}, CXR: {response_cxr.status_code}"

    # Parse JSON responses
    data_cx = response_cx.json()

    # Initialize info string
    info = ""
    success_count = 0
    max_articles = 20  # Attempt to fetch up to 40 articles in total

    # Step 2: Extract titles and links from both sources
    # Combine results from CX and CXR
    combined_data = []

    for item in data_cx.get("items", [])[:max_articles // 2]:
        combined_data.append({
            "title": item.get("title", "No Title"),
            "link": item.get("link", "No Link"),
        })

    # Step 3: Scrape articles from the combined results
    for article in combined_data:
        title = article["title"]
        link = article["link"]
        print(f"Scraping: {title} - {link}")

        try:
            # Fetch content from each URL
            article_response = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
            if article_response.status_code != 200:
                print(f"Failed to fetch {link}")
                continue

            # Parse the article content
            soup = BeautifulSoup(article_response.text, "html.parser")
            paragraphs = soup.find_all("p")  # Extract paragraphs
            article_text = clean_string(" ".join(p.get_text() for p in paragraphs))

            # Add to info string if content is found
            if article_text.strip():
                info += f"Title: {title}\nLink: {link}\nContent: {article_text[:1000]}...\n\n"  # Truncated for brevity
                success_count += 1
                print(f"Successfully scraped article: {title}")
        except Exception as e:
            print(f"Error scraping {link}: {e}")
            continue

    # Check if at least one article was scraped successfully
    if success_count == 0:
        return "Error: Unable to scrape any articles."

    print(f"Total successful articles: {success_count}")
    return info  # Return the concatenated string with article details


def  gather_information(topic, max_paragraphs=5):
    """ #12 default
    Fetch comprehensive information about the given topic by aggregating
    multiple search results, prioritizing recent news.

    :param topic: The topic to search for.
    :param max_paragraphs: Maximum number of paragraphs to gather.
    :return: Aggregated information as a single string.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    base_url = "https://www.google.com/search"
    news_url = f"{base_url}?q={topic.replace(' ', '+')}+2025"  # Use Google News search
    import requests


    query = f"Latest news on {topic} "
    # query = f"Latest news on {topic}  site:news.google.com"

    info = fetch_and_scrape_google_news(query)
    print(info)
    # Add Custom info:

    # Fetch from Wikipedia if not enough information
    if len(info.split()) < max_paragraphs * 30:
        print("Fetching additional information from Wikipedia...")
        wikipedia_url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        try:
            wiki_response = requests.get(wikipedia_url, headers=headers)
            if wiki_response.status_code == 200:
                wiki_soup = BeautifulSoup(wiki_response.text, "html.parser")
                wiki_paragraphs = wiki_soup.find_all("p")
                for p in wiki_paragraphs:
                    text = p.get_text().strip()
                    if text:
                        info += text + " "
                        if len(info.split()) >= max_paragraphs * 30:
                            break
            else:
                print(f"Failed to retrieve Wikipedia page for {topic}. Status code: {wiki_response.status_code}")
        except Exception as e:
            print(f"Error fetching Wikipedia results: {e}")

    # Fallback to DuckDuckGo
    if len(info.split()) < max_paragraphs * 30:
        print("Fetching additional information from DuckDuckGo...")
        try:
            with DDGS() as ddgs:
                results = ddgs.text(keywords=topic, max_results=20)
                for result in results:
                    snippet = result.get("body")
                    if snippet:
                        info += snippet + " "
                        if len(info.split()) >= max_paragraphs * 30:
                            break
        except Exception as e:
            print(f"Error fetching DuckDuckGo results: {e}")

    # Clean up the info text
    info = re.sub(r'\s+', ' ', info)
    print(f"Information gathered: {info[:200]}...")  # Print first 200 characters
    return info

import spacy


nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extract relevant entities and keywords from the given text using spaCy.
    """
    doc = nlp(text)
    entities = []

    # Extract named entities (e.g., PERSON, ORG, GPE, etc.)
    for ent in doc.ents:
        if ent.label_ in {"PERSON","ORG"}:  # Relevant entity types
            if ent.text.lower() not in (['skyler','theearliest','youtube']):
                entities.append(ent.text)


    return list(set(entities))
def extract_entity_photos(text):
    """
    Extract relevant entities and keywords from the given text using spaCy.
    """
    doc = nlp(text)
    entities = []

    # Extract named entities (e.g., PERSON, ORG, GPE, etc.)
    for ent in doc.ents:
        if ent.label_ in {"PERSON","ORG"}:  # Relevant entity types
            if ent.text.lower() not in (['skyler','theearliest','youtube']):
                entities.append(ent.text)

    return list(set(entities))[0]

# ------------------------
#  MAIN WORKFLOW
# ------------------------
def main():
    ###Configurations
    print("Fetching trending topics...")
    trending_topics = get_trending_topics()
    print(trending_topics)
    ### Add additional context with the prefix
    prefix = ''
    print(f"Prefix: {prefix}")


    subjects = [random.choice(trending_topics)]

    #### Manual subject override

    subjects = ["TinyZero Deep Seek UC berkeley"]  # Example subjects


    print(f"Selected subject: {subjects}")
    print("Gathering information...")
    info = ""
    for subject in subjects:
        info += gather_information(subject)
    print(f"Information gathered: {info[:200]}...")  # Displaying the first 200 characters
    print(f"Information length: {len(info)} characters")
    print("Generating SEO-optimized title and script...")
    topic_combination = " ".join(subjects)
    results = generate_title_and_tags.invoke({"topic": topic_combination, "info": info[:8000], "prefix": prefix})
    # Extracting content from results
    title_text = results["title"].content
    script_result = generate_script_search.invoke({"topic": topic_combination, "info": info[:8000], "title": title_text})

    script_text = script_result["script"].content
    tags = results["tags"].content
    description = results["desc"].content
    base_tags = subjects + ["News", "Trending"]
    generated_tags = [tag.strip() for tag in tags.split(',')]
    all_tags = list(set(base_tags + generated_tags))
    print(f"Script: {script_text}")

    print("Generating GIF search keywords...")
    print("Generating GIF search keywords...")
    search_term = f"{title_text} - {' '.join(subjects)}"
    entities = extract_entities(search_term)
    print(f"Extracted entities: {entities}")

    gif_term_results = generate_gif_search.invoke({"script": title_text})
    stock_results = generate_stock_search.invoke({"script": title_text})
    stock_subjects = stock_results["subjects"].content
    gif_terms = gif_term_results["tags"].content
    # gif_terms = ["geek","okay","The Matrix","Thanks!"]
    # gif_emotion = extract_entities(gif_terms)
    import json

    # Your string representation of an array

    # Convert the string to an actual list
    stock_subjects = json.loads(stock_subjects)

    # Verify the result


    print(f"Stock Subjects: {stock_subjects}")

    print(f"GIF Emotion: {gif_terms}")

    print(f"All Tags: {all_tags}")
    print(f"Generated Title: {title_text}")
    print(f"Generated Description: {description}")
    print(f"Generated Tags: {generated_tags}")

    # Create directory for assets
    video_directory = create_unique_directory(title_text)
    print(f"Assets will be saved in: {video_directory}")

    # Fetch GIFs
    print("Fetching GIFs from Giphy...")
    # gif_files = fetch_gifs_from_giphy([gif_terms], video_directory)

    gif_files = fetch_gifs_from_giphy(stock_subjects + subjects, video_directory)
    print(f"Fetched {len(gif_files)} GIFs.")
    stock_files = fetch_short_videos_from_pexels(stock_subjects,video_directory,max_duration=15)
    print(f"Fetched {len(stock_files)} Stock Files.")

    # Fetch images
    print("Fetching relevant images...")
    image_directory = fetch_relevant_images_and_videos(stock_subjects, video_directory)
    print(f"Images saved to {image_directory}\n")

    # Combine and shuffle media clips
    media_clips = stock_files + gif_files + [
        os.path.join(image_directory, img)
        for img in os.listdir(image_directory)
        if img.endswith((".jpg", ".png"))
    ]
    random.shuffle(media_clips)

    # Ensure the first clip is a GIF
    # first_gif = next((clip for clip in stock_files if clip in media_clips), None)
    # if first_gif:
    #     media_clips.remove(first_gif)
    #     media_clips.insert(0, first_gif)

    print(f"Total media clips: {len(media_clips)}")

    # Save the script to a file
    script_path = os.path.join(video_directory, "script.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_text)
    print(f"\nScript saved to {script_path}")

    # Generate audio from script text
    audio_path = os.path.join(video_directory, "audio.mp3")
    generate_audio(script_text, audio_path, selected_speaker='p243', emotion=gif_terms.strip())
    # generate_audio(script_text, audio_path, selected_speaker='p243', emotion=gif_emotion)
    print(f"Audio saved to {audio_path}")

    # Create the base video
    base_video_path = os.path.join(video_directory, "video_base.mp4")
    final_video_path = os.path.join(video_directory, "final_base.mp4")
    create_base_video_with_clips(audio_path, media_clips, base_video_path)
    print(f"Base video saved to {base_video_path}")
    #add_subtitles_to_video(base_video_path,final_video_path,CROWLS_INK_FONT_PATH)

    # Upload to YouTube
    print("Uploading video to YouTube...")
    upload_to_youtube(
        video_path=base_video_path,
        title=title_text,
        description=description,
        tags=all_tags,
        privacy_status="private"
    )
    print("Done!")

if __name__ == "__main__":
    main()
