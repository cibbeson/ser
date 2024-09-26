import tempfile
from ser.params import save_params, load_params, Params
from pathlib import Path
import shutil
from dataclasses import dataclass

@dataclass
class FakeParams:
    a: 1
    b: 2

def test_save_params():
    
    tmpdir = tempfile.mkdtemp()
    tmpdir = Path(tmpdir)
    p = Params('a',1,2,1.0,'b')
    save_params(tmpdir, p)
    params_output = load_params(tmpdir)
    shutil.rmtree(tmpdir)
    assert params_output == p

