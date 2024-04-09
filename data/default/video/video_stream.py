"""
TODO: Refactor in order to have a abstract video stream than can:
  - resize & cropping
  - stream splitting (virtual stream) so we can spawn multiple streams from same one
  - unify all video stream under this abstract video stream

"""
from core.data.default.video.video_stream_ffmpeg import VideoStreamFfmpegDataCapture as ParentDataCapture

_CONFIG = {
  **ParentDataCapture.CONFIG,

  'VALIDATION_RULES': {
    **ParentDataCapture.CONFIG['VALIDATION_RULES'],

  },
}


class VideoStreamDataCapture(ParentDataCapture):
  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    super(VideoStreamDataCapture, self).__init__(**kwargs)
    return
