"""Independent v1 receipt and request encoding for the test-only join ELF."""
import hashlib
from pathlib import Path
import struct
import sys

def digest(domain, body):
    return hashlib.sha256(b"PCI/v1/" + domain + b"\0" + body).digest()

def u32(n):
    return struct.pack("<I", n)

def ids(domain, values):
    raw = b"".join(map(u32, values))
    return digest(domain, struct.pack("<Q",len(values)) + raw)

def profile(t):
    import struct as s
    return s.pack("<8I", 1, 2560, 9728, 32, 8, 128, 151936, t) + s.pack("<ff", 1e6, 1e-6) + s.pack("<II", 0, 0) + bytes(16)

def fixture(p,n,directory):
    prompt = list(range(p))
    reply = list(range(1000,1000+n))
    checkpoint,suite,nonce=(bytes([i])*32 for i in (1,2,3))
    weights=[hashlib.sha256(b"w"+u32(i)).digest() for i in range(38)]
    model=digest(b"model/qwen3-4b/ordered-fp32-bf16", checkpoint+profile(0)+u32(38)+b"".join(weights))
    rq = digest(b"request/greedy-lowest-id", model+suite+nonce+ids(b"prompt",prompt)+u32(n)+u32(40960)+ids(b"stop-ids",[151645]))
    root=ids(b"reply",reply)
    data=bytearray(b"JOINTEST"+checkpoint+suite+b"".join(weights)+nonce+u32(p)+u32(n)+b"".join(map(u32,prompt+reply)))
    previous=ids(b"consumed",prompt+reply[:-1])
    for i in range(38):
        kind=0 if i==0 else 2 if i==37 else 1
        index=i-1 if kind==1 else 0
        output=root if i==37 else hashlib.sha256(u32(i)).digest()
        fields=u32(index)+u32(p+n-1)+u32(p-1 if i==37 else 0)+u32(n if i==37 else 0)+previous+output+weights[i]
        data.extend(u32(kind)+fields+digest(b"child",model+suite+rq+bytes([kind])+fields))
        previous=output
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"input").write_bytes(struct.pack("<Q",len(data))+data)
    (directory/"expected").write_bytes(b"JOINTEST"+rq+root)

if __name__=="__main__":
    fixture(int(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3]))
