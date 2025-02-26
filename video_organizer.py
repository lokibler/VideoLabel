import os
from pathlib import Path
from typing import List, Dict
import shutil
from datetime import datetime

import cv2
import numpy as np
from sklearn.cluster import KMeans
from pymediainfo import MediaInfo
import torch
from PIL import Image
from transformers import pipeline, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

class VideoProcessor:
    def __init__(self, input_folder: str, output_base: str):
        self.input_folder = Path(input_folder)
        self.output_base = Path(output_base)
        self.supported_formats = {'.mp4', '.mov', '.avi', '.mkv'}
        
        # Load CLIP model for image understanding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load image captioning model
        self.caption_generator = pipeline("image-to-text", 
                                       model="Salesforce/blip-image-captioning-base",
                                       device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Load sentence transformer for clustering
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

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
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                frames.append(pil_image)
        
        cap.release()
        return frames

    def get_video_description(self, video_path: Path) -> str:
        """Generate description using CLIP and BLIP."""
        frames = self.extract_frames(video_path)
        frame_descriptions = []
        
        # Get captions for each frame
        for frame in frames:
            caption = self.caption_generator(frame)[0]['generated_text']
            frame_descriptions.append(caption)
        
        # Combine frame descriptions into a coherent summary
        combined_desc = ' '.join(frame_descriptions)
        summary = f"This video shows {combined_desc.lower()}. The scenes appear to contain {frame_descriptions[-1].lower()}."
        
        return summary

    def embed_metadata(self, video_path: Path, description: str):
        """Embed description into video metadata using ffmpeg."""
        output_path = self.output_base / video_path.relative_to(self.input_folder)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use ffmpeg to embed metadata
        os.system(f'ffmpeg -i "{video_path}" -metadata description="{description}" '
                 f'-codec copy "{output_path}"')

    def cluster_videos(self, descriptions: Dict[Path, str], num_clusters: int = 5) -> Dict[Path, int]:
        """Cluster videos based on their descriptions using sentence embeddings."""
        video_paths = list(descriptions.keys())
        
        # Get embeddings using sentence transformer
        embeddings = self.sentence_transformer.encode(list(descriptions.values()))
        
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
    processor = VideoProcessor(
        input_folder="input_videos",
        output_base="processed_videos"
    )
    processor.process_videos()

if __name__ == "__main__":
    main() 