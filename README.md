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

- macOS 10.15+ (Catalina or newer)
- Miniforge or Miniconda for M1/M2 Macs (or Anaconda for Intel Macs)
- FFmpeg installed on your system
- Sufficient disk space for processed videos (~10GB for models)
- Internet connection for initial setup

## Installation

1. Install Miniforge/Miniconda (recommended for Mac):
```bash
# For Apple Silicon (M1/M2) Macs:
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o miniforge.sh

# For Intel Macs:
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o miniforge.sh

# Install Miniforge
bash miniforge.sh
```

2. Set up Conda environment:
```bash
# Create new conda environment
conda create -n video_organizer python=3.11

# Activate environment
conda activate video_organizer

# Verify Python installation
python --version
```

3. Clone and install dependencies:
```bash
# Clone repository
git clone https://github.com/yourusername/video_organizer.git
cd video_organizer

# Install dependencies using conda and pip
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

4. Install FFmpeg using Homebrew:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install FFmpeg
brew install ffmpeg
```

5. Set up Hugging Face access:
```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Install the Hugging Face CLI
pip install --upgrade huggingface_hub

# Log in to Hugging Face (you'll need to create a free account at huggingface.co)
huggingface-cli login
```

6. Download required models (this will take some time and ~10GB of disk space):
```bash
# Create and run the setup script
cat > setup_models.py << 'EOL'
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
EOL

# Run the setup script
python setup_models.py
```

7. Create directories and run:
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Create required directories
mkdir input_videos processed_videos

# Place your videos in input_videos directory

# Run the script
python video_organizer.py
```

## Environment Management

- To activate the environment:
```bash
conda activate video_organizer
```

- To deactivate when done:
```bash
conda deactivate
```

- To remove the environment (if needed):
```bash
conda remove --name video_organizer --all
```

## Usage

The script will process all video files in the `input_videos` directory and save the generated descriptions and clusters in the `processed_videos` directory.

### Supported Video Formats
- .mp4
- .mov
- .avi
- .mkv

### Processing Steps
0. Activate virtual environment: `source venv/bin/activate`
1. Place your videos in the `input_videos` directory
2. Run `python video_organizer.py`
3. Find processed videos in `processed_videos` directory, organized by content similarity
4. When done, deactivate virtual environment: `deactivate`

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
- All dependencies are isolated in the virtual environment

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

Mac-specific issues and solutions:

1. **Apple Silicon (M1/M2) compatibility**:
   - Use Miniforge instead of Anaconda for native ARM support
   - Ensure PyTorch is installed with proper architecture support
   - If using Rosetta 2, prefix commands with `arch -x86_64`

2. **Environment activation issues**:
   - Ensure Conda is initialized: `conda init zsh` (or `conda init bash`)
   - Restart terminal after initialization
   - Check active environment: `conda info --envs`

3. **FFmpeg installation**:
   - If Homebrew install fails, try: `brew update && brew upgrade`
   - Alternative install: `conda install ffmpeg -c conda-forge`

4. **Memory/Performance issues**:
   - For M1/M2 Macs, ensure using native ARM versions of packages
   - Monitor Activity Monitor for memory usage
   - Reduce batch size if needed

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.