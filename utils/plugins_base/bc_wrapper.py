from core.bc import DefaultBlockEngine
class BCWrapper:
  def __init__(self, blockchain_manager : DefaultBlockEngine):
    self.__bc : DefaultBlockEngine = blockchain_manager
    return
  
  @property
  def address(self):
    """
    Returns the address of the current node

    Returns
    -------
    str
        The address of the current node in the blockchain
    """
    return self.__bc.address
  
  def encrypt_str(self, str_data : str, str_recipient : str):
    """
    Encrypts a string using the public key of the recipient using asymmetric encryption

    Parameters
    ----------
    str_data : str
        the data to be encrypted (string)
    str_recipient : str
        the recipient's address (string) used as the public key

    Returns
    -------
    str
       the base64 encoded encrypted data
    """
    return self.__bc.encrypt(plaintext=str_data, recipient=str_recipient)
  
  def decrypt_str(self, str_b64data : str, str_sender : str):
    """
    Decrypts a base64 encoded string using the private key of the sender using asymmetric encryption

    Parameters
    ----------
    str_b64data : str
        The base64 encoded encrypted data
    str_sender : str
        The sender's address (string) used as the public key for decryption

    Returns
    -------
    str
       the decrypted data (string) that can be then decoded to the original data
    """
    return self.__bc.decrypt(encrypted_data_b64=str_b64data, sender=str_sender)
  
  
  def get_whitelist(self):
    """
    Returns the whitelist of the current node

    Returns
    -------
    list
        The list of addresses that are whitelisted
    """
    return self.__bc.whitelist
  
