class BasePlugin:
  def __init__(self, config, *args, **kwargs):
    super(BasePlugin, self).__init__(*args, **kwargs)
    self._on_init(config)
    return
  
  def _on_init(self, config):
    print("Base init requested")
    self.on_init(config)
    return

  def _on_command(self, data, **kwargs):
    return
  
  def on_command(self, data, **kwargs):
    self._on_command(data, **kwargs)
    return
  
  def process(self):
    print("Base process")
    return


class _MoxaMixinDeviceSetup:
  def __init__(self, *args, **kwargs):
    super(_MoxaMixinDeviceSetup, self).__init__(*args, **kwargs)
    self._moxa_device = None
    return
  
  def _update(self):
    print("Moxa update requested for {}".format(self._moxa_device))
    return
  
  
  def _moxa_device_factory(self, uri, **kwargs):
    print("Moxa device factory requested")
    self._moxa_device = uri
    return


class _MoxaMixinCommands:
  def __init__(self, *args, **kwargs):
    super(_MoxaMixinCommands, self).__init__(*args, **kwargs)
    return
  
  def __data_prep(self, data):
    dct_result = {}
    if isinstance(data, dict):
      dct_result = {k.lower(): v for k, v in data.items()}
    return dct_result
  
  def moxa_dispatch(self, data, **kwargs):
    dct_actions = self.__data_prep(data) 
    if "action" in dct_actions:
      func_name = "moxa_{}".format(dct_actions["action"])
      func = getattr(self, func_name, None)
      if func:
        func()
    
  def moxa_reboot(self):
    print(f"Moxa reboot requested on {self._moxa_device}")
    return
  
  def moxa_shutdown(self):
    print(f"Moxa shutdown requested on {self._moxa_device}")
    return
  
  def on_init(self):
    print(f"Moxa init requested on {self._moxa_device}")
    return


class BaseMoxaDriver(
  BasePlugin,
  _MoxaMixinCommands,
  _MoxaMixinDeviceSetup,
  ):
  def __init__(self, *args, **kwargs):
    super(BaseMoxaDriver, self).__init__(*args, **kwargs)
    return
  
  def on_init(self, config):
    self._moxa_device_factory(config["uri"])
    self._update()
    return
  
  
  def on_command(self, data, **kwargs):
    self.moxa_dispatch(data, **kwargs)
    return  

  def s1_moxa(self):
    raise NotImplementedError("s1_moxa not implemented")
  
  def s2_moxa(self):  
    raise NotImplementedError("s2_moxa not implemented")
  
  def s3_moxa(self):
    raise NotImplementedError("s3_moxa not implemented")
  
  def process(self):
    print("BaseMoxaDriver process for {}".format(self._moxa_device))
    
    self.process_section1()
    self.process_section2()
    self.process_section3()    
    return
  
  
  def process_section1(self):
    print(1)
    print(1)
    self.s1_moxa()
    return

  def process_section2(self):
    print(2)
    print(2)
    print(2)
    self.s2_moxa()
    return
  
  def process_section3(self):
    print(3)
    print(3)
    print(3)
    self.s3_moxa()
    return
  
  
  def s1_moxa(self):
    print("s1_moxa")
    return
  


###################

CONFIG = {
  "uri": "my_moxa_default"  
}
  
class Moxa1234(BaseMoxaDriver):
  
  def moxa_cool_update(self):
    print(f"Moxa cool update requested on {self._moxa_device}")
    return    
  
  # def s1_moxa(self):
  #   print("s1_moxa") # identical to with UserMoxa5467
  #   return
  
  def s2_moxa(self):  
    print("s2_moxa1234")
    return
  
  def s3_moxa(self):
    print("s3_moxa1234")
    return
  
class Moxa5467(BaseMoxaDriver):
  
  def moxa_cool_update(self):
    print(f"Moxa cool update requested on {self._moxa_device}")
    return    
  
  # def s1_moxa(self):
  #   print("s1_moxa") # identical to with UserMoxa1234
  #   return
  
  def s2_moxa(self):  
    print("s2_moxa5467")
    return
  
  def s3_moxa(self):
    print("s3_moxa5467")
    return  
  
class UserMoxa02(Moxa1234):
  def moxa_test(self):
    print("moxa_test")
    return

class UserMoxa03(Moxa5467):
  def __init__(self, config, *args, **kwargs):
    super(UserMoxa03, self).__init__(*args, **kwargs)
    return

  
  
if __name__ == '__main__':
  
  plg = UserMoxa02(CONFIG)
  plg.on_command({"action": "reboot"})
  plg.on_command({"action": "update"})
  plg.on_command({"action": "test"})
  
  plg.process()
  
  