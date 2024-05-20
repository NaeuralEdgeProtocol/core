from concurrent.futures import ThreadPoolExecutor
import os


class _ThreadingAPIMixin():
  def __init__(self):
    super(_ThreadingAPIMixin, self).__init__()
    return
  
  def threading_compute_in_parallel(self, func, lst_data, n_workers=None):
    if n_workers is None:
      n_workers = os.cpu_count() // 4
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
      results = executor.map(func, lst_data)
    return list(results)


  def threading_custom_code_compute_in_parallel(self, base64_code, lst_data, n_workers=None):
    base_custom_code_method, errors, warnings = self._get_method_from_custom_code(
      str_b64code=base64_code,
      self_var='plugin',
      method_arguments=['plugin', 'data']
    )

    custom_code_method = lambda data: base_custom_code_method(self, data)
    if errors is not None:
      errors_str = "\n".join([str(e) for e in errors])
      self.P(f"Errors found while getting custom code method:\n{errors_str}", color='r')
      return None
    
    warnings_str = "\n".join([str(w) for w in warnings])
    if len(warnings) > 0:
      self.P(f"Warnings found while getting custom code method:\n{warnings_str}", color='y')
    
    return self.threading_compute_in_parallel(custom_code_method, lst_data, n_workers)