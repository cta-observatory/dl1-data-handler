from dl1_data_handler.version import get_version
import json 
  
aux = get_version(pep440=False)
print(aux)
#details = {'version': aux}
