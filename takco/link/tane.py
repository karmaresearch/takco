import os, subprocess

class Tane():
    TMPNAME = 'tanefile.dat'
    class TaneException(Exception):
        pass
    
    def __init__(self, root, tmpdir = '/tmp/'):
        self.root = os.path.abspath(root)
        self.tanebin = os.path.join(self.root, 'bin', 'tane')
        self.tmpdir = tmpdir
        
    def rundf(self, df, level=None, num_records=None, num_attributes=None, tmpname=None, ):
        """Run TANE algorithm on dataframe to find functional dependencies."""
        nrows, ncols = df.shape
        level = level or ncols
        num_records = num_records or nrows
        num_attributes = num_attributes or ncols
        
        datfile = self._prepare(df, tmpname=tmpname)
        fds = self._run(datfile, level, num_records, num_attributes)
        
        # Aggregate dependents per determiner
        det_deps = {}
        for src, dst in fds:
            det = tuple(df.columns[list(src)])
            dep = df.columns[dst]
            det_deps.setdefault(det, set()).add(dep)
        return det_deps
        
    
    def _prepare(self, df, tmpname=None):
        # make dataframe categorical to get coded representation
        codes = df.astype('category').apply(lambda c: c.cat.codes)
        # put coded dataframe in temp directory
        fname = os.path.join(self.tmpdir, tmpname or self.TMPNAME)
        codes.to_csv(fname, header=False, index=False)
        return fname
        
    def _run(self, datfile, level, num_records, num_attributes):
        # run tane binary on coded data
        datfile = os.path.join(self.root, datfile)
        cmd = (self.tanebin, level, num_records, num_attributes, datfile)
        out = subprocess.run([str(arg) for arg in cmd], capture_output=True)
        if out.returncode:
            raise self.TaneException(out.stderr.decode())
        return list(self.parse_output(out.stdout.decode()))
    
    @staticmethod
    def parse_output(output):
        for line in output.splitlines():
            if line and line[0].isnumeric():
                try:
                    a, b = line.split('->', 1)
                    # TANE output is 1-indexed, so make 0-indexed
                    yield tuple(int(i)-1 for i in a.split()), int(b)-1
                except ValueError:
                    pass