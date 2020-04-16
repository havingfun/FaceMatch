from pytube import YouTube
import urllib.parse as urlparse
from urllib.parse import parse_qs

class YoutubeVideoReader():
    def __init__(self, url):
        self.url = url
        self.video_param = parse_qs(urlparse.urlparse(url).query)['v']
        self.video = None
        
    def download(self):
        if self.video is None:
            yt = YouTube(self.url)
            stream = yt.streams.filter(progressive = True, file_extension = "mp4").first()
            self.video = stream.download(f"video/{self.video_param[0]}")
        
    def get_video(self):
        return self.video