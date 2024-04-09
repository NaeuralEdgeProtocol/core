from core.utils.plugins_base.plugin_base_utils import _UtilsBaseMixin
from core.local_libraries.nn.th.utils import th_resize_with_pad

class _InferenceUtilsMixin(_UtilsBaseMixin):

  def __init__(self):
    super(_InferenceUtilsMixin, self).__init__()
    return
  
  
  def th_resize_with_pad(self,img, h, w, 
                         device=None, 
                         pad_post_normalize=False,
                         normalize=True,
                         sub_val=0, div_val=255,
                         half=False,
                         return_original=False,
                         normalize_callback=None,
                         fill_value=None,
                         return_pad=False,
                         **kwargs
                         ):
    return th_resize_with_pad(
      img, h, w, 
      device=device, 
      pad_post_normalize=pad_post_normalize,
      normalize=normalize,
      sub_val=sub_val, div_val=div_val,
      half=half,
      return_original=return_original,
      normalize_callback=normalize_callback,
      fill_value=fill_value,
      return_pad=return_pad,
      **kwargs
    )
  
