# Video Organizer

A Python tool that automatically processes video files, generates descriptions, and organizes them into content-based clusters. It uses CLIP for frame analysis and BLIP for generating natural language descriptions, which are then embedded into the video metadata. All processing is done locally without requiring any API calls.

## Features

- Automatically analyzes video content using CLIP
- Generates natural language descriptions using BLIP
- Embeds descriptions into video metadata
- Clusters similar videos into folders
- Supports multiple video formats (.mp4, .mov, .avi, .mkv)
- Works completely offline after initial model download

## Prerequisites

- Python 3.7+
- FFmpeg installed on your system
- Sufficient disk space for processed videos (~10GB for models)
- GPU recommended but not required
- Internet connection for initial setup

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video_organizer.git
cd video_organizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Hugging Face access:
```bash
# Install the Hugging Face CLI
pip install --upgrade huggingface_hub

# Log in to Hugging Face (you'll need to create a free account at huggingface.co)
huggingface-cli login
```

4. Download required models (this will take some time and ~10GB of disk space):
```python
# Run this Python script once to download models
from transformers import CLIPProcessor, CLIPModel, pipeline
from sentence_transformers import SentenceTransformer

# Download CLIP
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Download BLIP
pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Download Sentence Transformer
SentenceTransformer('all-MiniLM-L6-v2')
```

5. Install FFmpeg:
```bash
# On Ubuntu/Debian
sudo apt-get update && sudo apt-get install ffmpeg

# On macOS with Homebrew
brew install ffmpeg

# On Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

6. Create directories and run:
```bash
# Create required directories
mkdir input_videos processed_videos

# Place your videos in input_videos directory

# Run the script
python video_organizer.py
```

## Usage

The script will process all video files in the `input_videos` directory and save the generated descriptions and clusters in the `processed_videos` directory.

### Supported Video Formats
- .mp4
- .mov
- .avi
- .mkv

### Processing Steps
1. Place your videos in the `input_videos` directory
2. Run `python video_organizer.py`
3. Find processed videos in `processed_videos` directory, organized by content similarity

## Output

The script will create a folder for each cluster in the `processed_videos` directory. Each cluster folder will contain:

- Videos with similar content grouped together
- Each video will have its description embedded in its metadata

You can view the embedded descriptions using:
```bash
# On Linux/macOS
ffprobe -show_entries format_tags=description input.mp4

# Or using MediaInfo GUI on any platform
```

## Notes

- Uses CLIP for frame analysis and object detection
- Uses BLIP for generating natural language descriptions
- Uses KMeans clustering to group similar videos
- Embeds descriptions directly into video metadata using FFmpeg
- All processing is done locally on your machine
- First run will download several GB of models
- Processing speed depends on your CPU/GPU

## Configuration

You can modify these parameters in the script:
- Number of frames to analyze per video (`num_frames` in `extract_frames`)
- Number of clusters (`num_clusters` in `cluster_videos`)
- Supported video formats (`supported_formats` in `VideoProcessor`)

## Limitations

- Processing time depends on video length and quantity
- Quality of descriptions depends on the clarity of video frames
- Clustering effectiveness improves with larger video collections
- Requires sufficient memory for processing high-resolution videos
- Initial setup requires internet connection
- Large disk space needed for models (~10GB)

## Troubleshooting

Common issues and solutions:

1. **FFmpeg not found**: Ensure FFmpeg is installed and in your system PATH
2. **CUDA errors**: If using GPU, ensure PyTorch CUDA version matches your NVIDIA drivers
3. **Memory errors**: Reduce `num_frames` if processing high-resolution videos
4. **Model download errors**: 
   - Check your internet connection
   - Ensure you're logged in to Hugging Face (`huggingface-cli login`)
   - Try downloading models individually using the provided Python script
5. **Disk space errors**: Ensure you have ~10GB free for models
6. **CUDA out of memory**: Reduce batch size or use CPU by setting `device="cpu"`

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.