from moviepy import VideoFileClip

video = VideoFileClip("assets/reinforcement_learning_demo_video.mp4")

video = video.subclipped(0,20)

video.write_gif(
    "assets/atari_demo.gif"
)