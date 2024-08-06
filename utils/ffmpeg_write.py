"""
copy from moviepy
"""

import os
import subprocess as sp
from subprocess import DEVNULL
from typing import Optional


class FFMPEG_VideoWriter:
    def __init__(self, filename: str, audiofile: str,
                 width: int, height: int,
                 fps: int = 25, threads: int = 6,
                 logfile: Optional[str] = None):
        cmd = ['ffmpeg',
               '-y',
               '-loglevel',
               'error',
               '-f',
               'rawvideo',
               '-vcodec',
               'rawvideo',
               '-s',
               f'{width}x{height}',
               '-pix_fmt',
               'rgb24',
               '-r',
               str(fps),
               '-an',
               '-i',
               '-',
               '-i',
               audiofile,
               '-acodec',
               'aac',
               '-vcodec',
               'libx264',
               '-preset',
               'medium',
               '-threads',
               str(threads),
               '-pix_fmt',
               'yuv420p',
               filename]

        popen_params = {"stdout": DEVNULL,
                        "stderr": sp.PIPE if logfile is None else logfile,
                        "stdin": sp.PIPE}

        # This was added so that no extra unwanted window opens on windows
        # when the child process is created
        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

        self.filename = filename
        self.ext = self.filename.split(".")[-1]
        self.codec = 'libx264'
        self.proc = sp.Popen(cmd, **popen_params)

    def write_frame(self, img_array):
        """ Writes one frame in the file."""
        try:

            self.proc.stdin.write(img_array.tobytes())

        except IOError as err:
            _, ffmpeg_error = self.proc.communicate()
            error = (str(err) + ("\n\nMoviePy error: FFMPEG encountered "
                                 "the following error while writing file %s:"
                                 "\n\n %s" % (self.filename, str(ffmpeg_error))))

            if b"Unknown encoder" in ffmpeg_error:

                error = error + ("\n\nThe video export "
                                 "failed because FFMPEG didn't find the specified "
                                 "codec for video encoding (%s). Please install "
                                 "this codec or change the codec when calling "
                                 "write_videofile. For instance:\n"
                                 "  >>> clip.write_videofile('myvid.webm', codec='libvpx')") % self.codec

            elif b"incorrect codec parameters ?" in ffmpeg_error:

                error = error + ("\n\nThe video export "
                                 "failed, possibly because the codec specified for "
                                 "the video (%s) is not compatible with the given "
                                 "extension (%s). Please specify a valid 'codec' "
                                 "argument in write_videofile. This would be 'libx264' "
                                 "or 'mpeg4' for mp4, 'libtheora' for ogv, 'libvpx for webm. "
                                 "Another possible reason is that the audio codec was not "
                                 "compatible with the video codec. For instance the video "
                                 "extensions 'ogv' and 'webm' only allow 'libvorbis' (default) as a"
                                 "video codec."
                                 ) % (self.codec, self.ext)

            elif b"encoder setup failed" in ffmpeg_error:

                error = error + ("\n\nThe video export "
                                 "failed, possibly because the bitrate you specified "
                                 "was too high or too low for the video codec.")

            elif b"Invalid encoder type" in ffmpeg_error:

                error = error + ("\n\nThe video export failed because the codec "
                                 "or file extension you provided is not a video")

            raise IOError(error)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc.wait()

        self.proc = None

    # Support the Context Manager protocol, to ensure that resources are cleaned up.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


#
class FFMPEG_AudioWriter:

    def __init__(self, filename, fps, nbytes=2,
                 nchannels=2, codec='aac', bitrate=None,
                 input_video=None, logfile=None, ffmpeg_params=None):

        self.filename = filename
        self.codec = codec

        if logfile is None:
            logfile = sp.PIPE

        cmd = (["ffmpeg", '-y',
                "-loglevel", "error" if logfile == sp.PIPE else "info",
                "-f", 's%dle' % (8 * nbytes),
                "-acodec", 'pcm_s%dle' % (8 * nbytes),
                '-ar', "%d" % fps,
                '-ac', "%d" % nchannels,
                '-i', '-']
               + (['-vn'] if input_video is None else ["-i", input_video, '-vcodec', 'copy'])
               + ['-acodec', codec]
               + ['-ar', "%d" % fps]
               + ['-strict', '-2']  # needed to support codec 'aac'
               + (['-ab', bitrate] if (bitrate is not None) else [])
               + (ffmpeg_params if ffmpeg_params else [])
               + [filename])

        popen_params = {"stdout": DEVNULL,
                        "stderr": logfile,
                        "stdin": sp.PIPE}

        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000

        self.proc = sp.Popen(cmd, **popen_params)

    def write_frames(self, frames_array):
        try:

            self.proc.stdin.write(frames_array.tobytes())

        except IOError as err:
            ffmpeg_error = self.proc.stderr.read()
            error = (str(err) + ("\n\nMoviePy error: FFMPEG encountered "
                                 "the following error while writing file %s:" % self.filename
                                 + "\n\n" + str(ffmpeg_error)))

            if b"Unknown encoder" in ffmpeg_error:

                error = (error +
                         ("\n\nThe audio export failed because FFMPEG didn't "
                          "find the specified codec for audio encoding (%s). "
                          "Please install this codec or change the codec when "
                          "calling to_videofile or to_audiofile. For instance "
                          "for mp3:\n"
                          "   >>> to_videofile('myvid.mp4', audio_codec='libmp3lame')"
                          ) % (self.codec))

            elif b"incorrect codec parameters ?" in ffmpeg_error:

                error = (error +
                         ("\n\nThe audio export failed, possibly because the "
                          "codec specified for the video (%s) is not compatible"
                          " with the given extension (%s). Please specify a "
                          "valid 'codec' argument in to_videofile. This would "
                          "be 'libmp3lame' for mp3, 'libvorbis' for ogg...")
                         % (self.codec, self.ext))

            elif b"encoder setup failed" in ffmpeg_error:

                error = (error +
                         ("\n\nThe audio export failed, possily because the "
                          "bitrate you specified was two high or too low for "
                          "the video codec."))

            else:
                error = (error +
                         ("\n\nIn case it helps, make sure you are using a "
                          "recent version of FFMPEG (the versions in the "
                          "Ubuntu/Debian repos are deprecated)."))

            raise IOError(error)

    def close(self):
        if hasattr(self, 'proc') and self.proc:
            self.proc.stdin.close()
            self.proc.stdin = None
            if self.proc.stderr is not None:
                self.proc.stderr.close()
                self.proc.stdee = None
            # If this causes deadlocks, consider terminating instead.
            self.proc.wait()
            self.proc = None

    def __del__(self):
        # If the garbage collector comes, make sure the subprocess is terminated.
        self.close()

    # Support the Context Manager protocol, to ensure that resources are cleaned up.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
