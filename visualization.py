import numpy as np
from moviepy.editor import VideoClip, ImageSequenceClip, CompositeVideoClip, TextClip, concatenate_videoclips, \
    ImageClip, clips_array
from data_load import FeedingData, DriveDataSet


class Video(object):

    @staticmethod
    def from_generators(gif_file_name, feeding_data, feeding_angle, how_many_images_to_generate, image_generator):
        frames = []
        duration_pre_image = 0.5
        for index in range(how_many_images_to_generate):
            image, angle = image_generator(feeding_data)
            text = TextClip(txt="angle:{:.2f}/{:.2f}".format(feeding_angle, angle),
                            method="caption", align="North",
                            color="white", stroke_width=3, fontsize=18)
            text = text.set_duration(duration_pre_image)
            frames.append(CompositeVideoClip([
                ImageClip(image, duration=duration_pre_image),
                text
            ]))
        final = concatenate_videoclips(frames)
        final.write_gif(gif_file_name, fps=2)


    @staticmethod
    def from_udacity_sample_data(drive_data_set, file_name):
        dataset = drive_data_set
        frames = []
        duration_pre_image = 0.1
        total = len(dataset)
        for index in range(0, total, 4):
            print("working {}/{}".format(index + 1, total))
            record = dataset[index]
            image, angle = record.image(), record.steering_angle
            text = TextClip(txt="angle:{:.2f}".format(angle),
                            method="caption", align="North",
                            color="white", stroke_width=3, fontsize=18)
            text = text.set_duration(duration_pre_image)

            center_image_clip = ImageClip(image, duration=duration_pre_image)
            left_image_clip = ImageClip(record.left_image(), duration=duration_pre_image)
            right_image_clip = ImageClip(record.right_image(), duration=duration_pre_image)

            all_images_clip = clips_array([[left_image_clip, center_image_clip, right_image_clip]])
            frames.append(CompositeVideoClip([
                all_images_clip,
                text
            ]))
        final = concatenate_videoclips(frames)
        final.write_videofile(file_name, fps=10)
