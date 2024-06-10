from .version import get_version
import json 
  
aux = get_version(pep440=False)
print('aux')
print(aux)
details = {'version': aux}  
with open('testversion.py', 'w') as convert_file: 
     convert_file.write(json.dumps(details))

