class PostponedRequest:
  def __init__(self, solver_method, method_kwargs={}):
    self.__solver_method = solver_method.__name__
    self.__method_kwargs = method_kwargs
    return

  def get_solver_method(self, obj):
    # TODO: implement safer solution for this
    return getattr(obj, self.__solver_method)

  def get_method_kwargs(self):
    return self.__method_kwargs
