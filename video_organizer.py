import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import shutil
from datetime import datetime
import base64

from moviepy.editor import VideoFileClip
import cv2
from openai import OpenAI
import numpy as np
from sklearn.cluster import KMeans
from pymediainfo import MediaInfo
import torch
from torchvision import models, transforms
from PIL import Image

class VideoProcessor:
    def __init__(self, api_key: str, input_folder: str, output_base: str):
        self.client = OpenAI(api_key=api_key)
        self.input_folder = Path(input_folder)
        self.output_base = Path(output_base)
        self.supported_formats = {'.mp4', '.mov', '.avi', '.mkv'}
        
        # Load pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        
        # Load ImageNet class labels
        with open('imagenet_classes.txt', 'r') as f:
            self.categories = [s.strip() for s in f.readlines()]
            
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_frames(self, video_path: Path, num_frames: int = 3) -> List[Image.Image]:
        """Extract evenly spaced frames from video for analysis."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB and then to PIL Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                frames.append(pil_image)
        
        cap.release()
        return frames

    def analyze_frame(self, image: Image.Image) -> List[str]:
        """Get top predictions for a single frame using ResNet."""
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_batch)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_5_prob, top_5_catid = torch.topk(probabilities, 5)
        
        return [self.categories[idx] for idx in top_5_catid]

    def get_video_description(self, video_path: Path) -> str:
        """Generate a two-sentence description using frame analysis and GPT-3.5."""
        frames = self.extract_frames(video_path)
        frame_descriptions = []
        
        for frame in frames:
            top_predictions = self.analyze_frame(frame)
            frame_descriptions.extend(top_predictions)
        
        # Create a prompt for GPT-3.5
        prompt = (
            "Based on the following objects and scenes detected in video frames, "
            "write a concise two-sentence description of what might be happening in the video. "
            f"Detected elements: {', '.join(frame_descriptions)}."
        )
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )

        return response.choices[0].message.content

    def embed_metadata(self, video_path: Path, description: str):
        """Embed description into video metadata using ffmpeg."""
        output_path = self.output_base / video_path.relative_to(self.input_folder)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        metadata = {
            "description": description,
            "processed_date": datetime.now().isoformat()
        }
        
        # Use ffmpeg to embed metadata
        os.system(f'ffmpeg -i "{video_path}" -metadata description="{description}" '
                 f'-codec copy "{output_path}"')

    def cluster_videos(self, descriptions: Dict[Path, str], num_clusters: int = 5) -> Dict[Path, int]:
        """Cluster videos based on their descriptions using embeddings."""
        # Get embeddings for descriptions
        embeddings = []
        video_paths = list(descriptions.keys())
        
        for desc in descriptions.values():
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=desc
            )
            embeddings.append(response.data[0].embedding)

        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        return dict(zip(video_paths, clusters))

    def process_videos(self):
        """Main processing function."""
        descriptions = {}
        
        # Process each video
        for video_path in self.input_folder.rglob('*'):
            if video_path.suffix.lower() in self.supported_formats:
                print(f"Processing {video_path}")
                description = self.get_video_description(video_path)
                descriptions[video_path] = description
                self.embed_metadata(video_path, description)

        # Cluster videos and organize into folders
        clusters = self.cluster_videos(descriptions)
        
        # Move files to cluster folders
        for video_path, cluster_id in clusters.items():
            cluster_folder = self.output_base / f"cluster_{cluster_id}"
            cluster_folder.mkdir(parents=True, exist_ok=True)
            
            output_path = cluster_folder / video_path.name
            if not output_path.exists():
                shutil.copy2(video_path, output_path)

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    processor = VideoProcessor(
        api_key=api_key,
        input_folder="input_videos",
        output_base="processed_videos"
    )
    processor.process_videos()

if __name__ == "__main__":
    main() 