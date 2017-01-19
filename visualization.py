import numpy as np
from moviepy.editor import VideoClip, ImageSequenceClip, CompositeVideoClip, TextClip, concatenate_videoclips, ImageClip
from data_load import FeedingData


class Video(object):

    @staticmethod
    def from_generators(gif_file_name, feeding_data, feeding_angle, how_many_images_to_generate, image_generator):
        frames = []
        duration_pre_image = 1
        for index in range(how_many_images_to_generate):
            image, angle = image_generator(feeding_data)
            text = TextClip(txt="angle:{:.2f}/{:.2f}".format(feeding_angle, angle),
                            method="caption", align="North",
                            color="white", stroke_width=3)
            text = text.set_duration(duration_pre_image)
            frames.append(CompositeVideoClip([
                ImageClip(image, duration=duration_pre_image),
                text
            ]))
        final = concatenate_videoclips(frames)
        final.write_gif(gif_file_name, fps=1)


