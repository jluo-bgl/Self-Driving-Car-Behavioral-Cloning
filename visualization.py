import numpy as np
from moviepy.editor import VideoClip, ImageSequenceClip, CompositeVideoClip, TextClip, concatenate_videoclips, \
    ImageClip, clips_array
from data_load import FeedingData, DriveDataSet
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


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
        total = len(dataset.drive_records)
        for index in range(0, total, 4):
            print("working {}/{}".format(index + 1, total))
            record = dataset.drive_records[index]
            image, angle = record.image(), record.steering_angle
            text = TextClip(txt="angle:{:.2f}".format(angle),
                            method="caption", align="North",
                            color="white", stroke_width=3, fontsize=18)
            text = text.set_duration(duration_pre_image)

            center_image_clip = ImageClip(image, duration=duration_pre_image)
            left_image_clip = ImageClip(record.left_image(), duration=duration_pre_image)
            right_image_clip = ImageClip(record.right_image(), duration=duration_pre_image)

            all_images_clip = clips_array([[center_image_clip], [left_image_clip], [right_image_clip]])
            frames.append(CompositeVideoClip([
                all_images_clip,
                text
            ]))
        final = concatenate_videoclips(frames)
        final.write_videofile(file_name, fps=10)


class Plot(object):
    @staticmethod
    def angle_distribution(angles):
        mu = 0  # mean of distribution
        sigma = 1  # standard deviation of distribution
        x = angles

        num_bins = 80

        fig, ax = plt.subplots()

        # the histogram of the data
        n, bins, patches = ax.hist(x, num_bins, normed=1)

        # add a 'best fit' line
        y = mlab.normpdf(bins, mu, sigma)
        ax.plot(bins, y, '--')
        ax.set_xlabel('Angles')
        ax.set_ylabel('Probability density')
        ax.set_title('Histogram of Angles Sample Size {}'.format() + len(angles))

        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        return plt
