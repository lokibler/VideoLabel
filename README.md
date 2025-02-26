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
- Sufficient disk space for processed videos
- GPU recommended but not required

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

3. Run the script:
```bash
python video_organizer.py
```

## Usage

The script will process all video files in the `input_videos` directory and save the generated descriptions and clusters in the `processed_videos` directory.

1. Create input and output directories:
```bash
mkdir input_videos processed_videos
```

2. Place your video files in the `input_videos` directory

3. Download the ImageNet class labels:
```bash
wget https://raw.githubusercontent.com/pytorch/vision/master/torchvision/data/labels/imagenet_classes.txt
```

## Output

The script will create a folder for each cluster in the `processed_videos` directory. Each cluster folder will contain:

- Videos with similar content grouped together
- Each video will have its description embedded in its metadata

## Notes

- Uses CLIP for frame analysis and object detection
- Uses BLIP for generating natural language descriptions
- Uses KMeans clustering to group similar videos
- Embeds descriptions directly into video metadata using FFmpeg
- All processing is done locally on your machine

## Configuration

You can modify these parameters in the script:
- Number of frames to analyze per video (`num_frames` in `extract_frames`)
- Number of clusters (`num_clusters` in `cluster_videos`)
- Supported video formats (`supported_formats` in `VideoProcessor`)

## Cost Considerations

This script uses OpenAI's API for:
- GPT-3.5 text generation
- Text embeddings for clustering

Please be aware of the associated API costs when processing large numbers of videos.

## Limitations

- Processing time depends on video length and quantity
- Quality of descriptions depends on the clarity of video frames
- Clustering effectiveness improves with larger video collections
- Requires sufficient memory for processing high-resolution videos

## Troubleshooting

Common issues and solutions:

1. **FFmpeg not found**: Ensure FFmpeg is installed and in your system PATH
2. **CUDA errors**: If using GPU, ensure PyTorch CUDA version matches your NVIDIA drivers
3. **Memory errors**: Reduce `num_frames` if processing high-resolution videos
4. **API errors**: Verify your OpenAI API key and quota

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.