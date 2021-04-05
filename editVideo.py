from moviepy import editor as mp
import os
import numpy as np
import cv2

class AudioClip:
    def __init__(self, path):
        self.clip = mp.AudioFileClip(path)
        self.high_point_s = None
        self.title = None
        self.artist = None

    def set_high_point_s(self, high_point):
        self.high_point_s = high_point

    def set_title(self, title):
        self.title = title

    def set_artist(self, artist):
        self.artist = artist


class VideoCategory:
    going_to_plane = 0
    takeoff = 1
    out_window = 2
    freefall = 3
    tv_landing = 4
    tandem_landing = 5

class VideoClip:
    def __init__(self, path):
        self.clip = mp.VideoFileClip(path, audio=False)
        self.category = None
        self.high_point_s = None
        self.cut_s = None

    def set_category(self, category: VideoCategory):
        self.category = category

    def set_duration_s(self, duration_s):
        self.duration_s = duration_s

    def set_high_point_s(self, high_point_s):
        self.high_point_s = high_point_s

    def set_cut_time(self, cut_s):
        self.cut_s = cut_s


class TandemVideo:
    def __init__(self, render_file_path):
        self.render_file_path = render_file_path

        self.tandem_student = None
        self.tandem_instructor = None
        self.tandem_video = None
        self.drop_zone = None
        self.dropzone_www = None

        self.audio = None
        self.videos = []

    def load_audio(self, audio: AudioClip):
        self.audio = audio
        ## TODO Support multiple audio streams?

    def load_video(self, video_clip: VideoClip):
        self.videos.append(video_clip)

    def load_names(self, tandem_student, tandem_instructor, tandem_video, drop_zone, dropzone_www):
        self.tandem_student = tandem_student
        self.tandem_instructor = tandem_instructor
        self.tandem_video = tandem_video
        self.drop_zone = drop_zone
        self.dropzone_www = dropzone_www

    def make(self):
        w, h = self.videos[0].clip.w, self.videos[0].clip.h
        video_end = 1

        duration = 6
        if os.path.exists(self.drop_zone):
            logo = cv2.imread(self.drop_zone)
            logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
            img = np.ones([h, w, 3], dtype=np.uint8)

            scale_percent = 150
            width = int(logo.shape[1] * scale_percent / 100)
            height = int(logo.shape[0] * scale_percent / 100)
            logo = cv2.resize(logo, (width, height), interpolation=cv2.INTER_AREA)

            imx = int((w - width) / 2)
            imy = int((h - height) / 2)
            img[imy:imy+height, imx:imx+width, :] = logo
            dropzone_txt = mp.ImageClip(img)

        else:
            dropzone_txt = mp.TextClip(self.drop_zone, color="white", bg_color="black", fontsize=130, font='Ubuntu-Bold', size=(w, h))
        dropzone_txt = dropzone_txt.set_start(video_end).set_duration(duration).set_pos('center')
        dropzone_txt = mp.vfx.fadein(dropzone_txt, duration=1)
        dropzone_txt = mp.vfx.fadeout(dropzone_txt, duration=1)
        video_end += duration

        video_end += 1

        duration = 4
        presents_txt = mp.TextClip("Presenterer", color="white", bg_color="black", fontsize=130, font='Ubuntu-Bold', size=(w, h))
        presents_txt = presents_txt.set_start(video_end).set_duration(duration).set_pos('center')
        presents_txt = mp.vfx.fadein(presents_txt, duration=1)
        presents_txt = mp.vfx.fadeout(presents_txt, duration=1)
        video_end += duration

        video_end += 1

        duration = 5
        student_txt = mp.TextClip(self.tandem_student + "\n sitt tandemhopp!", color="white", bg_color="black", fontsize=130, font='Ubuntu-Bold', size=(w, h))
        student_txt = student_txt.set_start(video_end).set_duration(duration).set_pos('center')
        student_txt = mp.vfx.fadein(student_txt, duration=1)
        student_txt = mp.vfx.fadeout(student_txt, duration=1)
        video_end += duration

        intro = mp.CompositeVideoClip([dropzone_txt, presents_txt, student_txt])

        going_to_plane = None
        takeoff = None
        window = []
        freefall = None
        tv_landing = None
        tandem_landing = None

        for video in self.videos:
            if video.cut_s is not None:
                video.clip = video.clip.subclip(0, video.cut_s)
            if video.category == VideoCategory.going_to_plane:
                going_to_plane = video
            elif video.category == VideoCategory.takeoff:
                takeoff = video
            elif video.category == VideoCategory.out_window:
                window.append(video)
            elif video.category == VideoCategory.freefall:
                freefall = video
            elif video.category == VideoCategory.tv_landing:
                tv_landing = video
            elif video.category == VideoCategory.tandem_landing:
                tandem_landing = video


        ## Going to plane clip
        duration = going_to_plane.clip.duration
        if duration > 10: # TODO check this value
            going_to_plane.clip = going_to_plane.clip.subclip(duration-10, duration)
            duration = 10
        going_to_plane.clip = going_to_plane.clip.set_start(video_end)
        going_to_plane.clip = mp.vfx.fadein(going_to_plane.clip, duration=1)
        video_end += duration


        ## Take off clip
        duration = takeoff.clip.duration
        if duration > 15: # TODO check this value
            takeoff.clip = takeoff.clip.subclip(duration-15, duration)
            duration = 15
        takeoff.clip = takeoff.clip.set_start(video_end)
        video_end += duration

        ## Window ## TODO video of fun jumpers exits are not supported at all!
        duration = 0
        n_window_clips = len(window)
        for window_clip in window:
            duration += window_clip.clip.duration

        ## Todo this may fail when clips have varying lengts
        freefall_padding_s = 25

        if video_end + duration > self.audio.high_point_s - freefall_padding_s:
            duration = (self.audio.high_point_s - freefall_padding_s - video_end)/n_window_clips
            for window_clip in window:
                clip_duration = window_clip.clip.duration
                window_clip.clip = window_clip.clip.subclip(clip_duration-duration, clip_duration)

        window_clips = mp.concatenate_videoclips([window_clip.clip for window_clip in window])
        window_clips = window_clips.set_start(video_end)
        video_end += window_clips.duration


        ## Freefall:
        if freefall.high_point_s > freefall_padding_s:
            clip_duration = freefall.clip.duration
            freefall.clip = freefall.clip.subclip(freefall.high_point_s - freefall_padding_s, clip_duration)
            freefall.high_point_s = freefall_padding_s

        freefall.clip = freefall.clip.set_start(self.audio.high_point_s - freefall.high_point_s)
        video_end += freefall.clip.duration

        ## TV landing
        duration = 15
        old_duration = tv_landing.clip.duration
        if old_duration > duration:
            tv_landing.clip = tv_landing.clip.subclip(old_duration - duration, old_duration)

        tv_landing.clip = tv_landing.clip.set_start(video_end)
        tv_landing.clip = mp.vfx.fadeout(tv_landing.clip, duration=1)
        video_end += tv_landing.clip.duration

        ## Tandem landing
        credits_start_time = self.audio.clip.duration - 8
        prelanding_duration = 10
        duration = tandem_landing.clip.duration

        if tandem_landing.clip.duration < credits_start_time - video_end:
            credits_start_time = video_end + tandem_landing.clip.duration
        elif prelanding_duration > tandem_landing.high_point:
            tandem_landing.clip = tandem_landing.clip.subclip(tandem_landing.high_point-prelanding_duration, duration)
            tandem_landing.clip = tandem_landing.clip.subclip(0, credits_start_time - video_end)

        tandem_landing.clip = mp.vfx.fadein(tandem_landing.clip, duration=1)
        tandem_landing.clip = mp.vfx.fadeout(tandem_landing.clip, duration=1)
        tandem_landing.clip = tandem_landing.clip.set_start(video_end)
        video_end += tandem_landing.clip.duration

        video_end += 1

        # Credits
        credits_txt = "\n".join([
            "Tandeminstruktør:\n" + self.tandem_instructor,
            "\n\n",
            "Foto/Video:\n" + self.tandem_video,
            "\n\n",
            "Musikk:\n" + self.audio.artist + " - " + self.audio.title,
            "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
            self.dropzone_www
        ])

        duration = 13
        def position(t): ## TODO hard coded paramteres, get these from credits.w/h
            y = 1080-((1080+1100)/10*t)
            if y < -1100:
                y = -1100
            return ('center', y)

        credits = mp.TextClip(credits_txt, color="white", bg_color="black", fontsize=60, font="DejaVu Sans", method='label')
        credits = credits.set_duration(duration)
        credits = credits.set_position(position)
        credits = credits.set_start(credits_start_time+1)
        video_end += duration

        #Audio
        audio_clip = mp.CompositeAudioClip([self.audio.clip])
        audio_clip = audio_clip.set_start(0)
        audio_clip = audio_clip.set_duration(video_end)

        # Compose
        final_video = mp.CompositeVideoClip([intro, going_to_plane.clip, takeoff.clip, window_clips, freefall.clip, tv_landing.clip, tandem_landing.clip, credits])
        final_video = final_video.set_audio(audio_clip)
        #final_video = final_video.subclip(15, 20)
        #final_video = final_video.subclip(self.audio.high_point_s, self.audio.high_point_s+7)

        final_video.write_videofile(self.render_file_path, fps=30, bitrate='20000000', threads='4')


