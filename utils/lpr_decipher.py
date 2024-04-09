# -*- coding: utf-8 -*-

import string
import numpy as np

class LPRDecipher:
  SEPARATOR = '>'
  LST_CHARS = list(sorted(set(string.ascii_uppercase + string.digits + SEPARATOR)))
  DCT_CHARS = {str(k): v for k, v in zip(LST_CHARS, range(len(LST_CHARS)))}

  def post_process_softmaxes(self, softmaxes):
    lst_lps_decoded = [self.decode_lp(np.argmax(x, axis=-1).tolist()) for x in softmaxes]
    lps_decoded = [''.join(x) for x in lst_lps_decoded]
    return lps_decoded

  def decode_lp(self, lp):
    dct = {v: k for k, v in self.DCT_CHARS.items()}
    return [dct[c] for c in lp]

