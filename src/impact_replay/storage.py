"""Write completed binary payloads atomically, without in-place ZIP updates."""
import io,os,tempfile
from pathlib import Path
import numpy as np
import torch

def atomic_bytes(path,payload):
    path=Path(path)
    fd,tmp=tempfile.mkstemp(prefix='.writing-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(payload);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def save_npz(path,**arrays):
    b=io.BytesIO();np.savez_compressed(b,**arrays);atomic_bytes(path,b.getvalue())

def save_checkpoint(value,path):
    b=io.BytesIO();torch.save(value,b);atomic_bytes(path,b.getvalue())
