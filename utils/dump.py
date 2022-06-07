# Part of Deep Lab package
# Author: Hatef Monajemi (monajemi@stanford.edu)

import json
import hashlib
import pandas as pd

class DumpJSON():
    
    def __init__(self,
                 obj=None, name=None,
                 read_path='results.json', write_path='results.json'):
        if obj is not None:
            path = obj.anals_results_path+'/'+name+'.json'
            read_path=path
            write_path=path
        
        self.read_path=read_path
        self.write_path=write_path
        
        try:
            with open(self.read_path, 'r') as fp:
                self.results = json.load(fp)
        except:
                self.results = {}

    def count(self):
        return(len(self.results))

    def append(self, x):
        
        json_x = json.dumps(x)
        hash = hashlib.sha1(json_x.encode("UTF-8")).hexdigest()
        hash = hash[:10];   # take only the first 10. it s enough here
        tmp  = {hash:x}
        self.results.update(**tmp)
    
    def save_to_csv(self):
        self.save()
        self.to_csv()
        
    def save(self):
        with open(self.write_path, 'w') as fp:
            json.dump(self.results, fp)

    def to_csv(self):
        df = pd.DataFrame.from_dict(self.results)
        df = df.transpose()
        filename = self.write_path
        filename = filename.split('.')
        if len(filename)>1:
            filename[-1] = 'csv'

        filename = '.'.join(filename)
        df.to_csv(filename)
